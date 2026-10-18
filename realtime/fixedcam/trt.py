"""ONNX export and a static-batch TensorRT runner for the crop classifier.

The engine is built with a FIXED batch (pad to it): a static engine has one tactic set, constant
latency, and is what the Orin will run. Engines are tied to the GPU architecture and the TensorRT
version, so the ONNX is the portable artifact: rebuild the engine on the target with
``build_engine`` (or ``trtexec --onnx=... --fp16 --saveEngine=...``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .net import CROP


def export_onnx(model: torch.nn.Module, path: str | Path, batch: int = 32, opset: int = 17) -> None:
    model = model.eval().cpu()
    dummy = torch.zeros(batch, 4, CROP, CROP)
    kw = dict(input_names=["crops"], output_names=["logits", "offset"], opset_version=opset,
              do_constant_folding=True)
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(path), **kw)


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True,
                 workspace_mb: int = 256) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + "; ".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    Path(engine_path).write_bytes(bytes(blob))


class TRTClassifier:
    """callable: x [N, 4, 32, 32] float32 CUDA -> (probs [N, 3], offsets [N, 2]) float32 CUDA."""

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.buf = {}
        for n in names:
            shape = tuple(self.engine.get_tensor_shape(n))
            dt = self.engine.get_tensor_dtype(n)
            tdt = torch.float16 if dt == trt.DataType.HALF else torch.float32
            self.buf[n] = torch.zeros(shape, dtype=tdt, device="cuda")
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.batch = self.buf[self.inp].shape[0]

    def __call__(self, x: torch.Tensor):
        probs, offs = [], []
        stream = torch.cuda.current_stream().cuda_stream
        for i in range(0, x.shape[0], self.batch):
            chunk = x[i:i + self.batch]
            n = chunk.shape[0]
            inp = self.buf[self.inp]
            inp[:n].copy_(chunk)
            if n < self.batch:
                inp[n:].zero_()
            self.ctx.execute_async_v3(stream)
            probs.append(torch.softmax(self.buf["logits"][:n].float(), dim=1))
            offs.append(self.buf["offset"][:n].float().clone())
        if not probs:
            dev = x.device
            return torch.zeros((0, 3), device=dev), torch.zeros((0, 2), device=dev)
        return torch.cat(probs), torch.cat(offs)


class TorchClassifier:
    """Same interface, plain PyTorch (development / training-time evaluation)."""

    def __init__(self, model: torch.nn.Module, half: bool = False):
        self.model = model.eval().cuda()
        self.half = half
        if half:
            self.model.half()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        if x.shape[0] == 0:
            return torch.zeros((0, 3), device=x.device), torch.zeros((0, 2), device=x.device)
        logits, off = self.model(x.half() if self.half else x)
        return torch.softmax(logits.float(), dim=1), off.float()


class EnsembleClassifier:
    """Average of several classifiers (e.g. engines trained with different seeds). The net is
    ~0.3 ms per batch on a laptop GPU, so a small ensemble costs little and removes seed noise."""

    def __init__(self, members):
        self.members = list(members)

    def __call__(self, x: torch.Tensor):
        outs = [m(x) for m in self.members]
        probs = torch.stack([o[0] for o in outs]).mean(0)
        offs = torch.stack([o[1] for o in outs]).mean(0)
        return probs, offs
