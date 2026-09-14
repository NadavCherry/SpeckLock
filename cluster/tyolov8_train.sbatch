#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-10:00:00
#SBATCH --job-name sl_tytrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tytrain-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=15200M
#SBATCH --array=0-5
### 15200M, measured: the two ARD-MAV seeds that were previously OOM-killed at a 10,000M
### ceiling (0, 1) were retrained under the `plots: False` fix below and both completed
### cleanly this time -- seed 0 peaked at 12,663 MiB (57 epochs, early-stopped), seed 1 at
### 10,289 MiB (61 epochs). The fix worked: with `plots: True`, ultralytics regenerates its
### per-epoch matplotlib figures every validation pass, and RAM had climbed roughly linearly
### with epoch count (NPS seeds measured uncapped at 6,011 MiB/50 epochs and 8,029 MiB/57,
### ~288 MiB/epoch, before the fix); nothing in this repository reads results.png/
### confusion_matrix.png/PR_curve.png/labels.jpg (checked with a repo-wide grep). 15200M is
### the worse of the two real peaks (12,663 MiB) plus ~20%.
### GPU memory peaked at 4,456 MiB (nvidia-smi, CUDA context included) on the original pilot.
### 10 h: the pilot's NPS epochs took ~130 s with validation, so ARD-MAV's 70 (1.9x the tiles)
### are ~4.8 h; about twice that. The array throttle (%N) is set at submission from
### cluster/gpu_budget.sh, because the limit of 7 GPUs is shared with every job this account runs.
###
### Temporal-YOLOv8, reimplemented (configs/experiments/prior_art.py): 70 epochs, seeds 0-2 on
### each benchmark, into work/runs_tyolov8/ -- a root of its own, so no other arm's run can be
### overwritten. Paired against temporal_<ds> at 100 epochs (work/runs_e100/).
###
### NO --requeue. tools/train.py refuses to start over a recorded run without --force, so a
### requeued task would fail rather than restart from epoch 0 under the same name; a preempted
### task is resubmitted by hand, visibly. Pass FORCE=1 (--export) to retry a seed that already
### has a MANIFEST.json from a failed attempt -- unset by default so an accidental resubmission
### of a GOOD run aborts loudly instead of silently overwriting it.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT

I=$SLURM_ARRAY_TASK_ID
DS=ardmav; [ "$I" -ge 3 ] && DS=nps
SEED=$(( I % 3 )); CFG="tyolov8_${DS}"
RUN="work/runs_tyolov8/${CFG}-s${SEED}"
echo "JOB $SLURM_JOB_ID task $I = $CFG seed $SEED, 70 epochs -> $RUN on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

/usr/bin/time -v python -u tools/train.py --config "$CFG" --seed "$SEED" \
    --out-root work/runs_tyolov8 --vram-gib 24 ${FORCE:+--force}
rc=$?

[ -f "$RUN/weights/best.pt" ] && ls -la "$RUN/weights/best.pt" || echo "*** NO WEIGHTS ***"
python -c "import json; r = json.load(open('$RUN/RESULT.json')); print('status:', r.get('status'), '| duration_s:', r.get('duration_s'), '| vram:', r.get('vram'))" 2>/dev/null
cg=$(awk -F: '$2 ~ /memory/ || $2 == "" {print $3; exit}' /proc/self/cgroup)
for d in "$cg" "$(dirname "$cg")"; do
  for f in "/sys/fs/cgroup/memory$d/memory.max_usage_in_bytes" "/sys/fs/cgroup$d/memory.peak"; do
    [ -r "$f" ] && echo "cgroup memory peak: $(( $(cat "$f") / 1048576 )) MiB  ($f)"
  done
done
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "TYOLOV8 TRAIN OK (${CFG}-s${SEED})" || echo "TYOLOV8 TRAIN FAILED (${CFG}-s${SEED})"
exit $rc
