#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name sl_tybuild
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/tybuild-%j.out
#SBATCH --mem=800M
### 800M is an ESTIMATE: the builder is one process decoding 1920x1080 video into a 31-frame
### grayscale ring buffer (~62 MB) and writing JPEG tiles. Each build runs under /usr/bin/time -v,
### and one process's peak is the job's, so the next edit to this line should come from the log.
###
### NO --gpus: fetching weights, decoding video and writing tiles are infrastructure.
###
### Builds the Temporal-YOLOv8 datasets (configs/experiments/prior_art.py): taps t-15, t, t+15,
### no stabilisation, and otherwise the same splits, stride, tiles and labels as the paired
### SpeckLock arms. The names carry _dt15_centred_staboff, so no shipped dataset can be touched.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
python -m pytest -q --no-header -x dronedet/tests/test_centred_taps.py 2>&1 | tail -2
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "*** centred-tap tests failed -- not building ***"; exit 1; }

### 1. YOLOv8m's COCO weights, fetched on a CPU node, never on a GPU node.
[ -f yolov8m.pt ] || python -c "from ultralytics.utils.downloads import attempt_download_asset as a; print(a('yolov8m.pt'))"
python -c "import os, sys; s = os.path.getsize('yolov8m.pt'); print('yolov8m.pt:', s, 'bytes'); sys.exit(0 if s > 40e6 else 1)" \
    || { echo "*** yolov8m.pt missing or truncated ***"; exit 1; }

### 2. The paper samples "around 15 frames" either side at 30 fps: about half a second. Every
###    video's rate is recorded, and a clip the BUILDER reads (training / validation) outside
###    27-33 fps (+/-10 %) stops the build, since training on it would change what the window
###    means. The history, so the tolerance is not mistaken for one chosen after the fact:
###      - job 21175147, tolerance 29-31, train/val only: stopped by NPS Clip_017 at 28.0 fps,
###        where 15 frames is 0.536 s -- inside the paper's "around". Widened to 27-33.
###      - job 21175224, test clips added: stopped by two NPS TEST clips at 59.9 fps, where 15
###        frames is 0.25 s. The builder never reads a test clip, so they cannot change what
###        the arm is trained on; they are named below and scored as the paired SpeckLock arm
###        already is on them, with fixed frame offsets (its 6 frames are 0.10 s there). How
###        they are reported is declared in configs/experiments/prior_art.py.
###    A video that cannot be found fails the check; it used to be skipped.
python - <<'PY' || { echo "*** frame-rate check failed -- not building ***"; exit 1; }
import collections, importlib.util, sys
import cv2
spec = importlib.util.spec_from_file_location("mde", "tools/make_dataset_external.py")
M = importlib.util.module_from_spec(spec); sys.modules["mde"] = M; spec.loader.exec_module(M)
bad = []
for name, ids, find, test in (
        ("ardmav", M._ard_all(), lambda v: M.ARD_ROOT / "videos" / f"{v}.mp4", set(M.ARD_TEST_IDS)),
        ("nps", [*M.NPS_TRAIN, *M.NPS_VAL, *M.NPS_TEST], M._nps_video, set(M.NPS_TEST))):
    fps, missing = {}, []
    for v in ids:
        p = find(v)
        if p is None or not p.exists():
            missing.append(v)
            continue
        c = cv2.VideoCapture(str(p)); fps[v] = round(c.get(cv2.CAP_PROP_FPS), 2); c.release()
    hist = ", ".join(f"{k} fps x{n}" for k, n in sorted(collections.Counter(fps.values()).items()))
    print(f"  {name}: {len(fps)} videos ({hist}); 15 frames = "
          f"{15 / max(fps.values()):.3f} .. {15 / min(fps.values()):.3f} s")
    odd = sorted(v for v, f in fps.items() if not 29.5 <= f <= 30.5)
    if odd:
        print("    outside 29.5-30.5 fps: "
              + ", ".join(f"{v} {fps[v]}{' (TEST)' if v in test else ''}" for v in odd))
    if missing:
        print(f"    MISSING: {', '.join(missing)}")
        bad.append(name)
    built = [f for v, f in fps.items() if v not in test]
    print(f"    clips the builder reads: {len(built)}, fps {min(built)} .. {max(built)}")
    if min(built) < 27.0 or max(built) > 33.0:
        bad.append(name)
sys.exit(1 if bad else 0)
PY

### 3. The two builds -- the shipped arms' flags, plus the window and the stabiliser.
rc=0
for task in ardmav-temporal-tiled nps-temporal-tiled; do
  echo "########## $task"
  /usr/bin/time -v python -u tools/make_dataset_external.py --task "$task" --tile 640 \
      --stride-train 4 --stride-val 10 --min-side 0 --dt 15 --taps centred --stab off || rc=1
done
for d in ardmav_yolo_temporal_dt15_centred_staboff nps_yolo_temporal_dt15_centred_staboff; do
  echo "=== $d"; cat "work/ext_datasets/$d/BUILD.json" 2>/dev/null
  for s in train val; do echo "  $s: $(ls "work/ext_datasets/$d/images/$s" 2>/dev/null | wc -l) tiles"; done
done
echo "=== RESULT ==="; echo "exit=$rc"; exit $rc
