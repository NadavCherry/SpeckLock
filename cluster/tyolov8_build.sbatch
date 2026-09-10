#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name sl_tybuild
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tybuild-%j.out
#SBATCH --mem=800M
### 800M is an ESTIMATE: the builder is one process decoding 1920x1080 video into a 31-frame
### grayscale ring buffer (~62 MB) and writing JPEG tiles. Each build runs under /usr/bin/time -v,
### and one process's peak is the job's, so the next edit to this line should come from the log.
###
### NO --gpus: fetching weights, decoding video and writing tiles are infrastructure.
###
### Builds the Temporal-YOLOv8 datasets (configs/experiments/prior_art.py): taps t-15, t, t+15,
### no stabilisation, and otherwise the same splits, stride, tiles and labels as the paired
### SpeckLock arms. The names carry _dt15_centred_staboff, so no shipped dataset can be touched.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
python -m pytest -q --no-header -x dronedet/tests/test_centred_taps.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "*** centred-tap tests failed -- not building ***"; exit 1; }

### 1. YOLOv8m's COCO weights, fetched on a CPU node, never on a GPU node.
[ -f yolov8m.pt ] || python -c "from ultralytics.utils.downloads import attempt_download_asset as a; print(a('yolov8m.pt'))"
python -c "import os, sys; s = os.path.getsize('yolov8m.pt'); print('yolov8m.pt:', s, 'bytes'); sys.exit(0 if s > 40e6 else 1)" \
    || { echo "*** yolov8m.pt missing or truncated ***"; exit 1; }

### 2. The paper's window is half a second, which is 15 frames only at ~30 fps. Refuse to build
###    from videos that are not, rather than silently changing what "15" means.
python - <<'PY' || { echo "*** frame-rate check failed -- not building ***"; exit 1; }
import importlib.util, sys
import cv2
spec = importlib.util.spec_from_file_location("mde", "tools/make_dataset_external.py")
M = importlib.util.module_from_spec(spec); sys.modules["mde"] = M; spec.loader.exec_module(M)
bad = []
for name, paths in (("ardmav", [M.ARD_ROOT / "videos" / f"{v}.mp4" for v in M._ard_all()]),
                    ("nps", [M._nps_video(c) for c in list(M.NPS_TRAIN) + list(M.NPS_VAL)])):
    fps = []
    for p in paths:
        if p is None:
            continue
        c = cv2.VideoCapture(str(p)); fps.append(round(c.get(cv2.CAP_PROP_FPS), 2)); c.release()
    print(f"  {name}: {len(fps)} videos, fps {min(fps)} .. {max(fps)}")
    if min(fps) < 29.0 or max(fps) > 31.0:
        bad.append(name)
sys.exit(1 if bad else 0)
PY

### 3. The two builds -- the shipped arms' flags, plus the window and the stabiliser.
rc=0
for task in ardmav-temporal-tiled nps-temporal-tiled; do
  echo "########## $task"
  /usr/bin/time -v python -u tools/make_dataset_external.py --task "$task" --tile 640 \
      --stride-train 4 --stride-val 10 --min-side 0 --dt 15 --taps centred --stab off || rc=1
done
for d in ardmav_yolo_temporal_dt15_centred_staboff nps_yolo_temporal_dt15_centred_staboff; do
  echo "=== $d"; cat "work/ext_datasets/$d/BUILD.json" 2>/dev/null
  for s in train val; do echo "  $s: $(ls "work/ext_datasets/$d/images/$s" 2>/dev/null | wc -l) tiles"; done
done
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
