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
### A regeneration may add columns or prose; it may not move a cell in a column the committed
### report already has. tools/report_diff.py checks exactly that, and names any cell that moved.
### (Its predecessor compared every numeral in order: right for a prose-only regeneration,
### wrong the moment a column is added on purpose -- and it had already failed one correct
### report, job 21172331, by counting the "1" in "best-F1".)
git show HEAD:work/ablation/REPORT.md > "report_head.$$.md" || exit 1
python tools/report_diff.py "report_head.$$.md" work/ablation/REPORT.md
rc=$?; rm -f "report_head.$$.md"
[ $rc -eq 0 ] || { echo "*** A SHARED CELL CHANGED -- not a column-additive regeneration"; exit 1; }
git diff --stat -- work/ablation/REPORT.md
echo "=== RESULT ==="
