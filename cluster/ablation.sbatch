#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name sl_ablation
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/ablation-%j.out
#SBATCH --mem=60M
### 60M, measured: the first run (job 21172331) peaked at 48,176 kB under /usr/bin/time --
### that plus ~20%. It was first submitted at 250M on a reasoned estimate ("numpy-only
### imports, eight small JSON files, roughly 200 MB"), which was 5x too high. Reasoning about
### memory over-asks; the step below keeps printing its own peak so this line stays measured.
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
### Digits glued to letters are words, not numbers: "F1", "1a", "1b". The first version of
### this check counted them, and failed a correct regeneration whose new footnote says
### "best-F1" (job 21172331: 91 numerals before, 92 after -- the extra one was the F1).
num = lambda t: re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", t)
assert num(new) == num(old), f"{len(num(old))} numerals before, {len(num(new))} after"
print(f"VERIFIED: all {len(num(new))} numerals identical to HEAD -- prose-only regeneration")
PY
git diff --stat -- work/ablation/REPORT.md
echo "=== RESULT ==="
