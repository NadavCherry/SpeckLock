#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-10:00:00
#SBATCH --job-name sl_tytrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tytrain-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=10000M
#SBATCH --array=0-5
### 10000M, measured: the pilot (job 21175865) training this arm on NPS at batch 8 peaked at
### 8,362 MiB for the whole job -- the cgroup's figure, which counts the four data-loader workers
### that /usr/bin/time cannot see (its single-process peak was 4,930 MiB). Plus ~20 %. No image
### cache is used (tools/train.py passes none), so ARD-MAV's larger training set adds label
### arrays, not images. GPU memory peaked at 4,456 MiB (nvidia-smi, CUDA context included).
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
### task is resubmitted by hand, visibly.
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
    --out-root work/runs_tyolov8 --vram-gib 24
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
