#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_tylocal
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tylocal-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=12G
#SBATCH --array=0-2
### --mem ABOVE IS NOT THE REQUEST: the submit command passes the pilot's measured cgroup peak
### plus ~20 %, as for the training array, and says which number it used.
###
### Temporal-YOLOv8 on the project's own task. Fine-tune from ITS OWN NPS checkpoint, exactly as
### cluster/local_finetune.sbatch does for ours and YOLOMG, then detect on the held-out 10_06
### flight with the flags cluster/local_eval_ft.sbatch used for our arms (--conf 0.001 --batch 1)
### plus this arm's window. Both steps are GPU work; scoring is CPU work and runs afterwards in
### cluster/tyolov8_local_score.sbatch.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT

S=$SLURM_ARRAY_TASK_ID
INIT="$ROOT/work/runs_tyolov8/tyolov8_nps-s${S}/weights/best.pt"
W="$ROOT/work/runs_ft/tyolov8_local_ab-s${S}/weights/best.pt"
echo "JOB $SLURM_JOB_ID task $S = tyolov8_local_ab seed $S, initialised from tyolov8_nps-s${S}, on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$INIT" ] || { echo "*** no init weights at $INIT"; exit 1; }

/usr/bin/time -v python -u tools/train.py --config tyolov8_local_ab --seed "$S" \
    --weights "$INIT" --out-root work/runs_ft --vram-gib 24 || { echo "*** fine-tune failed"; exit 1; }
[ -f "$W" ] || { echo "*** no weights at $W"; exit 1; }

/usr/bin/time -v python -u tools/infer_tiled.py --weights "$W" --mode temporal \
    --dt 15 --taps centred --stab off \
    --gt-dir "$ROOT/work/ext_datasets/gt/local" --out-dir "$ROOT/work/det/local_ft/tyolov8_local_ab-s${S}" \
    --video-root "$ROOT/data/videos" --conf 0.001 --batch 1 --device 0
rc=$?
cg=$(awk -F: '$2 ~ /memory/ || $2 == "" {print $3; exit}' /proc/self/cgroup)
for d in "$cg" "$(dirname "$cg")"; do
  for f in "/sys/fs/cgroup/memory$d/memory.max_usage_in_bytes" "/sys/fs/cgroup$d/memory.peak"; do
    [ -r "$f" ] && echo "cgroup memory peak: $(( $(cat "$f") / 1048576 )) MiB  ($f)"
  done
done
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "TYOLOV8 LOCAL OK (seed $S)" || echo "TYOLOV8 LOCAL FAILED (seed $S)"
exit $rc
