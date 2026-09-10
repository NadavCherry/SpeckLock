#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_tyinfer
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tyinfer-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=6000M
#SBATCH --array=0-5
### INFERENCE ONLY. Scoring (tools/evaluate.py, tools/check_scorecard.py) is CPU work and runs
### afterwards in cluster/tyolov8_score.sbatch, so no GPU sits idle through it.
###
### 6000M: the pilot's largest single process (job 21175865, /usr/bin/time) peaked at 4,930 MiB
### -- the trainer's main process, holding the same network and CUDA context inference needs --
### plus ~20 %. Inference is one process, so /usr/bin/time below measures it exactly, and the
### next edit of this line comes from that.
###
### Flags: exactly those cluster/eval_e100.sbatch scored the paired arm with (--conf 0.001,
### --batch 1, default tile / overlap / merge distance / max-det), plus this arm's window:
### --dt 15 --taps centred --stab off, which runs the builder's own generator, so the arm is
### scored on the representation it was trained on.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT

I=$SLURM_ARRAY_TASK_ID
DS=ardmav; [ "$I" -ge 3 ] && DS=nps
SEED=$(( I % 3 ))
TAG="tyolov8_${DS}-e70-s${SEED}"
if [ "$DS" = ardmav ]; then VIDROOT="$ROOT/data/external/ard_mav/ARD-MAV/videos"
else VIDROOT="$ROOT/data/external/nps/Videos_x/Videos"; fi
GT="$ROOT/work/ext_datasets/gt/$DS"
W="$ROOT/work/runs_tyolov8/tyolov8_${DS}-s${SEED}/weights/best.pt"
DET="$ROOT/work/det/$DS/$TAG"

echo "JOB $SLURM_JOB_ID task $I = $TAG (centred, unstabilised, dt 15) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }

/usr/bin/time -v python -u tools/infer_tiled.py --weights "$W" --mode temporal \
    --dt 15 --taps centred --stab off \
    --gt-dir "$GT" --out-dir "$DET" --video-root "$VIDROOT" \
    --conf 0.001 --batch 1 --device 0
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "TYOLOV8 INFER OK ($TAG)" || echo "TYOLOV8 INFER FAILED ($TAG)"
exit $rc
