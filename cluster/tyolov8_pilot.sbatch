#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:40:00
#SBATCH --job-name sl_typilot
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/typilot-%j.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=12G
### A MEASUREMENT RUN: the Temporal-YOLOv8 NPS arm, trained for 20 minutes and then stopped, so
### the real training jobs can ask for what they need. No GPU job in this project had ever
### recorded its memory -- the 24G in matched100.sbatch is a habit, not a number -- and
### /usr/bin/time under-counts a trainer, because the data-loader workers are separate processes
### whose RSS it does not sum. So this also reads the job's cgroup peak, which covers every
### process in the job, and the training array asks for that plus ~20%. 12G is the estimate
### this run exists to replace.
###
### WHY A TIME LIMIT, NOT --epochs 1. The config validator refuses any run shorter than
### close_mosaic (12 epochs) -- "mosaic would never be on" -- and that is what stopped the first
### pilot (job 21175380) three seconds in. 13 epochs is the shortest run it accepts; `timeout`
### ends it after 20 minutes, by which point it has trained with mosaic on and validated, so
### memory has peaked. A run stopped that way writes no RESULT.json, so GPU memory is sampled
### with nvidia-smi instead. Its weights are discarded (work/runs_pilot).
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 15 > "vram.$$" 2>/dev/null &
SMI=$!
/usr/bin/time -v timeout --signal=TERM 20m python -u tools/train.py --config tyolov8_nps --seed 0 \
    --epochs 13 --out-root work/runs_pilot --vram-gib 24 --force
rc=$?
kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null
echo "GPU memory.used peak (nvidia-smi, 15 s samples): $(sort -n "vram.$$" | tail -1) MiB"; rm -f "vram.$$"
R=work/runs_pilot/tyolov8_nps-s0/results.csv
[ -f "$R" ] && { echo "epochs completed (results.csv):"; cut -d, -f1-2 "$R"; }
cg=$(awk -F: '$2 ~ /memory/ || $2 == "" {print $3; exit}' /proc/self/cgroup)
for d in "$cg" "$(dirname "$cg")"; do
  for f in "/sys/fs/cgroup/memory$d/memory.max_usage_in_bytes" "/sys/fs/cgroup$d/memory.peak"; do
    [ -r "$f" ] && echo "cgroup memory peak: $(( $(cat "$f") / 1048576 )) MiB  ($f)"
  done
done
### 124 is timeout ending the run on schedule -- the plan. Anything else is a failure.
[ "$rc" -eq 124 ] && { echo "stopped after 20 min, as planned"; rc=0; }
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
