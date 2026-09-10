#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name sl_cammotion
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/cammotion-%j.out
#SBATCH --mem=500M
### 500M is an ESTIMATE: one process holding two full-resolution float32 frames, a Hanning
### window and phaseCorrelate's FFT buffers -- a few hundred MB with cv2 and numpy loaded.
### Each step runs under /usr/bin/time -v, and a single process's peak RSS is the job's, so
### the next edit to this line should come from that number.
###
### NO --gpus: phase correlation on the CPU, over videos that already exist.
###
### Camera motion, measured identically on every dataset the README makes a claim about:
### the 15 ARD-MAV and 10 NPS test videos, and the project's own two videos.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
python -m pytest -q --no-header -x dronedet/tests/test_camera_motion.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "*** camera_motion tests failed ***"; exit 1; }
rc=0
/usr/bin/time -v python -u tools/camera_motion.py --dataset ardmav \
    --gt-dir work/ext_datasets/gt/ardmav --video-root data/external/ard_mav/ARD-MAV/videos || rc=1
/usr/bin/time -v python -u tools/camera_motion.py --dataset nps \
    --gt-dir work/ext_datasets/gt/nps --video-root data/external/nps/Videos_x/Videos || rc=1
/usr/bin/time -v python -u tools/camera_motion.py --dataset local \
    --pair data/videos/10_06.mp4 realtime/work/gt_1006_v2.json \
    --pair data/videos/07_05.mp4 work/gt_user.json || rc=1
echo "=== RESULT ==="; ls -la work/reports/camera_motion/; echo "exit=$rc"; exit $rc
