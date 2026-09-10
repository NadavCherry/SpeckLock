#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name sl_ablation
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/ablation-%j.out
#SBATCH --mem=250M
### 250M is an ESTIMATE, not a measurement: this generator had never run under
### /usr/bin/time. Its import chain is numpy-only (dronedet.metrics / detections / gt; no cv2
### or torch), it reads eight small tracked JSON files, and it block-bootstraps a few hundred
### boxes -- roughly 200 MB, plus ~20%. The step below prints its own peak RSS; the next edit
### to this line should come from that number, not from this paragraph.
###
### NO --gpus: it reads detection JSON that already exists and does arithmetic.
###
### Regenerates work/ablation/REPORT.md from its generator. It is the only artefact that
### produces the headline 0.159 -> 0.895, so prose fixed in the README and not here would be
### the pattern this repository keeps repeating: corrected where found, standing where repeated.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
/usr/bin/time -v python -u tools/ablation_temporal.py --out work/ablation/REPORT.md > /dev/null
rc=$?
echo "ablation_temporal exit=$rc"
[ $rc -eq 0 ] || exit $rc
### Prose and column labels may change; no number may. Every numeral in the regenerated report
### must equal, in order, every numeral in the committed one.
python - <<'PY' || { echo "*** A NUMBER CHANGED -- this is not a prose-only regeneration"; exit 1; }
import re, subprocess
new = open("work/ablation/REPORT.md", encoding="utf-8").read()
old = subprocess.run(["git", "show", "HEAD:work/ablation/REPORT.md"], capture_output=True,
                     text=True, check=True).stdout
num = lambda t: re.findall(r"\d+(?:\.\d+)?", t)
assert num(new) == num(old), f"{len(num(old))} numerals before, {len(num(new))} after"
print(f"VERIFIED: all {len(num(new))} numerals identical to HEAD -- prose-only regeneration")
PY
git diff --stat -- work/ablation/REPORT.md
echo "=== RESULT ==="
