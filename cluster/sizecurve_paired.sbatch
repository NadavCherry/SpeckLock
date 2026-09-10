#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:30:00
#SBATCH --job-name sl_scpaired
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/scpaired-%j.out
#SBATCH --mem=750M
### Only the size curves that carry a PAIRED table -- the `mission` bins of the datasets
### with more than one test sequence -- re-run so their verdicts go through the Holm
### correction (docs/research/INFRA.md section 6.1). The aitod bin set and local_ft carry
### no paired table; re-running them would spend an hour to change nothing.
###
### Sized from the last measured run, cluster_logs/sizecurve-20894039.out (/usr/bin/time):
### nps/mission 40:00 at 608 MB, ardmav/mission 1:23:00 at 525 MB. 750M is the larger peak
### plus ~20% (submit ardmav with --mem=650M); 2h30 is the longer run plus a margin.
### JobAcctGatherType=(null) here, so the step prints its own peak RSS for next time.
###
### NO --gpus: this reads detection JSON that already exists and does arithmetic.
###
###     sbatch cluster/sizecurve_paired.sbatch nps
###     sbatch --mem=650M cluster/sizecurve_paired.sbatch ardmav
set -uo pipefail
DS=${1:?usage: sizecurve_paired.sbatch nps|ardmav}
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname): $DS / mission, root $ROOT"

### Gate on the suite: the paired block runs last, after the whole resampling, so a
### regression there would otherwise surface an hour in.
python -m pytest -q --no-header -x 2>&1 | tail -5
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "*** SUITE FAILED -- refusing to produce numbers from a broken evaluator ***"; exit 1
fi

case "$DS" in nps|ardmav) ;; *) echo "unknown dataset $DS"; exit 2 ;; esac
GT=work/ext_datasets/gt/$DS
### Arms in exactly the order cluster/sizecurve.sbatch passes them: the paired test pairs
### runs by position, so the order IS the seed matching.
ARGS=()
for s in 0 1 2; do ARGS+=(--arm ours        "$GT" "work/det/$DS/temporal_${DS}-e100-s$s"); done
for s in 0 1 2; do ARGS+=(--arm yolomg      "$GT" "work/det/$DS/yolomg_${DS}_seed$s"); done
for s in 0 1 2; do ARGS+=(--arm ours-single "$GT" "work/det/$DS/singleframe_${DS}-e100-s$s"); done

/usr/bin/time -v python -u tools/size_curve.py --dataset "$DS" --bins mission \
    --rule iou --iou 0.5 --conf 0.001 --resamples 1000 \
    "${ARGS[@]}" --out work/reports/size_curve
rc=$?
echo "size_curve exit=$rc"
[ $rc -eq 0 ] || exit $rc

### The rerun must be a pure re-verdict: every number the correction does not touch has to
### equal what is committed at HEAD, or this is a different experiment, not a correction.
python - "$DS" <<'PY' || { echo "*** VERIFICATION FAILED: not a pure re-verdict of HEAD"; exit 1; }
import json, subprocess, sys
ds = sys.argv[1]
path = f"work/reports/size_curve/{ds}_mission.json"
new = json.load(open(path, encoding="utf-8"))
old = json.loads(subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True,
                                text=True, check=True).stdout)
assert new["arms"] == old["arms"], "per-bin results drifted from HEAD"
keep = ("bin", "seed", "d_ap", "lo", "hi", "p_perm", "favours")
assert [{k: r[k] for k in keep} for r in new["paired"]] == \
       [{k: r[k] for k in keep} for r in old["paired"]], "paired effects drifted from HEAD"
assert [r["significant_raw"] for r in new["paired"]] == [r["significant"] for r in old["paired"]]
gone = [f"{r['bin']} s{r['seed']}" for r in new["paired"]
        if r["significant_raw"] and not r["significant"]]
print(f"VERIFIED {ds}: arms and paired effects identical to HEAD; "
      f"Holm removed {len(gone)} mark(s): {gone}")
PY
echo "=== RESULT ==="
