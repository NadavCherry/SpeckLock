#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_typilot
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/typilot-%j.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=12G
### A MEASUREMENT RUN: one epoch of the Temporal-YOLOv8 NPS arm, so the real training jobs can
### ask for what they need. No GPU job in this project had ever recorded its memory -- the 24G
### in matched100.sbatch is a habit, not a number -- and /usr/bin/time under-counts a trainer,
### because the data-loader workers are separate processes whose RSS it does not sum. So this
### also reads the job's cgroup peak, which covers every process in the job, and the training
### array asks for that plus ~20%. 12G is the estimate this run exists to replace.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

/usr/bin/time -v python -u tools/train.py --config tyolov8_nps --seed 0 --epochs 1 \
    --out-root work/runs_pilot --vram-gib 24 --force
rc=$?

python -c "import json; r = json.load(open('work/runs_pilot/tyolov8_nps-s0/RESULT.json')); print('peak VRAM (torch):', r.get('peak_vram_gib'), 'GiB; strides:', r.get('strides'))" 2>/dev/null
cg=$(awk -F: '$2 ~ /memory/ || $2 == "" {print $3; exit}' /proc/self/cgroup)
for d in "$cg" "$(dirname "$cg")"; do
  for f in "/sys/fs/cgroup/memory$d/memory.max_usage_in_bytes" "/sys/fs/cgroup$d/memory.peak"; do
    [ -r "$f" ] && echo "cgroup memory peak: $(( $(cat "$f") / 1048576 )) MiB  ($f)"
  done
done
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
