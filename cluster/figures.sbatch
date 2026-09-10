#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_figures
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/figures-%j.out
#SBATCH --mem=450M
### 450M, measured: the last run (cluster_logs/figures-20944249.out) peaked at 367 MB for
### all five paper figures, video decode included -- that plus ~20%. It was 1G, sized from
### an earlier 215 MB measurement; the peak moved, which is why every step prints its own.
### The first pass at right-sizing cut this from 12G to 3G by reasoning about what
### matplotlib "should" need; the instrumented run showed 3G was still 14x over.
### This cluster sets JobAcctGatherType=(null), so sacct MaxRSS and sstat are both
### empty and nothing ever contradicted an over-request. Each python step below runs
### under /usr/bin/time -v and prints its own peak RSS into this job's log.

### NO --gpus. Every figure is read from a saved result file; fig5 decodes video and
### rebuilds the temporal stack on the CPU. Nothing here benefits from CUDA.
###
### Regenerates EVERY figure the repository ships, not only the paper set. The social card,
### the three architecture drawings, the result charts and the gallery page carry numbers too, and the social
### card spent weeks showing claims the README had already retracted because nothing ever
### re-rendered it. PNGs come from cairosvg when present, otherwise rsvg-convert.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
rc=0
for step in "tools/make_paper_figures.py --results work --out docs/media/paper" \
            "tools/make_social_card.py" \
            "tools/make_arch_figures_final.py" \
            "tools/make_arch_figure_system.py" \
            "tools/make_result_charts.py" \
            "tools/make_gallery.py"; do
    echo "########## $step"
    /usr/bin/time -v python -u $step || { echo "*** FAILED: $step"; rc=1; }
done
echo "=== RESULT ==="; ls -la docs/media/paper/ docs/media/*.png 2>/dev/null; echo "exit=$rc"; exit $rc
