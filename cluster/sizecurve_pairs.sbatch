#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:30:00
#SBATCH --job-name sl_scpairs
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/scpairs-%j.out
#SBATCH --mem=880M
### Size curves for two comparisons the shipped size-crossover table does not make.
### tools/size_curve.py tests its first "our" arm against the FIRST other arm only, so each
### comparison is its own two-arm run, into its own directory: the shipped
### work/reports/size_curve/<ds>_mission.* is never written by this job.
###
###   single   ours (temporal, 100 ep) - ours single-frame (100 ep). ADDED AFTER the overall
###            comparison was known to be "no difference" on both benchmarks. The bins are the
###            shipped `mission` bins, fixed long before; the table is its own Holm family and
###            is reported whatever it shows.
###   tyolov8  ours (temporal, 100 ep) - Temporal-YOLOv8 (70 ep, configs/experiments/prior_art.py).
###            Declared before any of that arm's scorecards existed.
###
### 880M: cluster/sizecurve_paired.sbatch's measured suite-gate peak (743,428 kB) + ~20 %. The
### size_curve step reads two arms here, fewer than the three it was measured with.
### NO --gpus: this reads detection JSON that already exists and does arithmetic.
###
###     sbatch cluster/sizecurve_pairs.sbatch ardmav single
###     sbatch cluster/sizecurve_pairs.sbatch nps tyolov8
set -uo pipefail
DS=${1:?usage: sizecurve_pairs.sbatch nps|ardmav single|tyolov8}
PAIR=${2:?usage: sizecurve_pairs.sbatch nps|ardmav single|tyolov8}
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname): $DS / mission, ours vs $PAIR, root $ROOT"

/usr/bin/time -v -o "suite_time.$$" python -m pytest -q --no-header -x 2>&1 | tail -5
rc=${PIPESTATUS[0]}
grep -E "Elapsed \(wall|Maximum resident" "suite_time.$$" | sed 's/^/suite gate: /'; rm -f "suite_time.$$"
[ "$rc" -eq 0 ] || { echo "*** SUITE FAILED -- refusing to produce numbers from a broken evaluator ***"; exit 1; }

case "$DS" in nps|ardmav) ;; *) echo "unknown dataset $DS"; exit 2 ;; esac
GT=work/ext_datasets/gt/$DS
### Seeds in the same order for both arms: the paired test matches runs by position.
ARGS=()
for s in 0 1 2; do ARGS+=(--arm ours "$GT" "work/det/$DS/temporal_${DS}-e100-s$s"); done
case "$PAIR" in
  single)  for s in 0 1 2; do ARGS+=(--arm ours-single "$GT" "work/det/$DS/singleframe_${DS}-e100-s$s"); done ;;
  tyolov8) for s in 0 1 2; do ARGS+=(--arm tyolov8 "$GT" "work/det/$DS/tyolov8_${DS}-e70-s$s"); done ;;
  *) echo "unknown pair $PAIR"; exit 2 ;;
esac
OUT=work/reports/size_curve/vs_$PAIR

/usr/bin/time -v python -u tools/size_curve.py --dataset "$DS" --bins mission \
    --rule iou --iou 0.5 --conf 0.001 --resamples 1000 \
    "${ARGS[@]}" --out "$OUT"
rc=$?
echo "size_curve exit=$rc"
[ $rc -eq 0 ] || exit $rc

### Same detections, same evaluator, same resampling seed, same position in the arm list: the
### ours column must equal the shipped curve exactly, and a single-frame column's per-seed APs
### must equal the shipped ones. Anything else means this is not the comparison it claims to be.
python - "$DS" "$OUT" "$PAIR" <<'PY' || { echo "*** VERIFICATION FAILED"; exit 1; }
import json, sys
ds, out, pair = sys.argv[1:4]
new = json.load(open(f"{out}/{ds}_mission.json", encoding="utf-8"))
old = json.load(open(f"work/reports/size_curve/{ds}_mission.json", encoding="utf-8"))
assert new["arms"]["ours"] == old["arms"]["ours"], "ours drifted from the shipped curve"
if pair == "single":
    for b, c in new["arms"]["ours-single"]["bins"].items():
        assert c["seeds"] == old["arms"]["ours-single"]["bins"][b]["seeds"], f"ours-single {b} drifted"
print(f"VERIFIED {ds}/{pair}: shipped columns reproduced exactly")
PY
echo "=== RESULT ==="; echo "exit=0"
