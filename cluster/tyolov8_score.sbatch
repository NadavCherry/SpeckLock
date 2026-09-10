#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_tyscore
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tyscore-%j.out
#SBATCH --mem=850M
### 850M is an ESTIMATE: tools/size_curve.py, which loads the same ground truth and three arms'
### detections at once, peaked at 624,708 kB (NPS) and 511,676 kB (ARD-MAV); one arm's scoring
### should need no more. Every step runs under /usr/bin/time -v, one process each, so the log
### holds the measured peak for the next edit of this line.
###
### NO --gpus: scoring is CPU work. It runs after cluster/tyolov8_infer.sbatch, with the same
### evaluator, flags and output naming as cluster/eval_e100.sbatch used for the paired arm.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"

rc=0
for DS in ardmav nps; do
  EXTRA=""; [ "$DS" = ardmav ] && EXTRA="--official-split"
  GT="$ROOT/work/ext_datasets/gt/$DS"
  for SEED in 0 1 2; do
    TAG="tyolov8_${DS}-e70-s${SEED}"
    W="$ROOT/work/runs_tyolov8/tyolov8_${DS}-s${SEED}/weights/best.pt"
    DET="$ROOT/work/det/$DS/$TAG"
    echo "########## $TAG"
    if [ ! -d "$DET" ]; then echo "*** NO DETECTIONS at $DET"; rc=1; continue; fi
    /usr/bin/time -v python -u tools/evaluate.py --dataset "$DS" --model "$TAG" $EXTRA \
        --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
        --out "$ROOT/work/scorecards/${TAG}.json" || { rc=1; continue; }
    python tools/check_scorecard.py "$ROOT/work/scorecards/${TAG}.json" || rc=1
  done
done
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
