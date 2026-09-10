#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_tylocbuild
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tylocbuild-%j.out
#SBATCH --mem=300M
### 300M is sized from a measurement plus one unmeasured step. The same builder peaked at
### 222,352 kB building ARD-MAV's 1920x1080 centred set (job 21175379, /usr/bin/time); these
### videos are 1280x720. The test gate below imports the same modules plus pytest and has never
### been measured, so the estimate for the job is ~250 MB and this is that plus ~20 %. Both
### steps now run under /usr/bin/time, so the next edit of this line comes from the log.
###
### NO --gpus: decoding video and writing tiles are infrastructure.
###
### The local dataset (train 07_05, test 10_06) for tyolov8_local_ab: taps t-15, t, t+15, no
### stabilisation, and otherwise the shipped local arms' flags (configs/experiments/local_video.py).
### Its directory carries _dt15_centred_staboff, so no shipped local dataset can be touched.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
/usr/bin/time -v -o "gate_time.$$" python -m pytest -q --no-header -x dronedet/tests/test_centred_taps.py \
    dronedet/tests/test_builder_passes_window.py dronedet/tests/test_prior_art_local.py 2>&1 | tail -2
g=${PIPESTATUS[0]}
grep -E "Maximum resident" "gate_time.$$" | sed 's/^/test gate: /'; rm -f "gate_time.$$"
[ "$g" -eq 0 ] || { echo "*** tests failed -- not building ***"; exit 1; }

/usr/bin/time -v python -u tools/make_dataset_external.py --task local-temporal --tile 640 \
    --stride-train 1 --stride-val 4 --min-side 0 --dt 15 --taps centred --stab off
rc=$?
D=work/ext_datasets/local_yolo_temporal_dt15_centred_staboff
echo "=== $D"; cat "$D/BUILD.json" 2>/dev/null
for s in train val; do echo "  $s: $(ls "$D/images/$s" 2>/dev/null | wc -l) tiles"; done
[ -f work/ext_datasets/gt/local/10_06.json ] || { echo "*** no 10_06 test ground truth"; rc=1; }
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
