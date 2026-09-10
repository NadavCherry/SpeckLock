#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_tylocscore
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tylocscore-%j.out
#SBATCH --mem=250M
### 250M is an ESTIMATE: tools/ablation_temporal.py, which scores four arms on the same 10_06
### ground truth, peaked at 48,560 kB (cluster/ablation.sbatch); a block bootstrap over ~12
### blocks adds little. Each step runs under /usr/bin/time -v, one process, so the log holds
### the measurement for the next edit of this line.
###
### NO --gpus: scoring is arithmetic on detections that already exist.
###
### Temporal-YOLOv8 on the project's own task, scored exactly as cluster/local_eval_ft.sbatch
### scored the three shipped arms -- same ground truth, protocol, block length and resample
### count -- with the new arm added as a fourth. Each seed's report goes to a NEW file; the
### shipped local_ft_10_06_seed<S>.md is never written. tools/report_diff.py then requires every
### cell the two reports share to be unchanged: the three shipped arms must come out exactly as
### committed, or this is not the comparison it claims to be.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export PYTHONPATH=$ROOT
### The shipped arms' detections live only in the main checkout; they are read, never written.
SHIP=${SHIPPED_DETS:-/home/cherryn/projects/SpeckLock/work/det/local_ft}
GT="$ROOT/work/ext_datasets/gt/local/10_06.json"
echo "JOB $SLURM_JOB_ID on $(hostname); shipped detections from $SHIP"

rc=0
for S in 0 1 2; do
  NEW="$ROOT/work/det/local_ft/tyolov8_local_ab-s${S}"
  [ -d "$NEW" ] || { echo "*** no detections at $NEW"; rc=1; continue; }
  OUT="work/reports/local_ft_tyolov8_10_06_seed${S}.md"
  echo "############ seed $S"
  /usr/bin/time -v python -u tools/compare_single_video.py --gt "$GT" --baseline singleframe \
      --dets singleframe="$SHIP/singleframe_local_ab-s${S}" \
             temporal="$SHIP/temporal_local_ab-s${S}" \
             yolomg="$SHIP/yolomg_local_ft_seed${S}" \
             tyolov8="$NEW" \
      --protocol specklock-centre --block-frames 30 --out "$OUT" || { rc=1; continue; }
  python tools/report_diff.py "work/reports/local_ft_10_06_seed${S}.md" "$OUT" || rc=1
  ### The direct question -- ours against Temporal-YOLOv8 -- with Temporal-YOLOv8 as the
  ### baseline, so the temporal row's delta, interval and p are ours minus theirs. The report
  ### above answers a different question (each arm against the single frame). Declared before
  ### any of this arm's detections existed.
  /usr/bin/time -v python -u tools/compare_single_video.py --gt "$GT" --baseline tyolov8 \
      --dets temporal="$SHIP/temporal_local_ab-s${S}" tyolov8="$NEW" \
      --protocol specklock-centre --block-frames 30 \
      --out "work/reports/local_ft_ours_vs_tyolov8_10_06_seed${S}.md" || rc=1
done
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
