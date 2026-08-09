
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from suop_parsers import parse_xyz, parse_case_settings, parse_head_info, parse_ping_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPO = str(PROJECT_ROOT / "suop_repo" / "SUOP_dataset")
CASE_LIST = str(DATA_DIR / "case_paths_120.txt")
OUT_DIR = DATA_DIR / "cache"
os.makedirs(OUT_DIR, exist_ok=True)

with open(CASE_LIST) as f:
    case_paths = [l.strip() for l in f if l.strip()]

manifest = []
t0 = time.time()
for i, cp in enumerate(case_paths):
    case_dir = os.path.join(REPO, cp)
    out_name = cp.replace("/", "__")
    out_npz = os.path.join(OUT_DIR, out_name + ".npz")
    try:
        xyz = parse_xyz(os.path.join(case_dir, "point_cloud.xyz"))
        cs = parse_case_settings(os.path.join(case_dir, "metadata", "case_settings.txt"))
        hi = parse_head_info(os.path.join(case_dir, "metadata", "head_info.txt"))
        pi = parse_ping_info(os.path.join(case_dir, "metadata", "ping_info.csv"))

        np.savez_compressed(
            out_npz,
            xyz=xyz.astype(np.float32),
            n_points=xyz.shape[0],
        )
        meta = {
            "case_path": cp,
            "n_points": int(xyz.shape[0]),
            "x_min": float(xyz[:,0].min()), "x_max": float(xyz[:,0].max()),
            "y_min": float(xyz[:,1].min()), "y_max": float(xyz[:,1].max()),
            "z_min": float(xyz[:,2].min()), "z_max": float(xyz[:,2].max()),
            "intensity_min": float(xyz[:,3].min()), "intensity_max": float(xyz[:,3].max()),
            "sound_velocity": cs.get("SoundVelocity"),
            "range_threshold": cs.get("RangeThreshold"),
            "ping_count": hi.get("PingCount"),
            "status": "ok",
        }
    except Exception as e:
        meta = {"case_path": cp, "status": "error", "error": str(e)}
    manifest.append(meta)
    if (i+1) % 10 == 0:
        print(f"[{i+1}/{len(case_paths)}] elapsed={time.time()-t0:.1f}s", flush=True)

with open(os.path.join(OUT_DIR, "cache_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

n_ok = sum(1 for m in manifest if m["status"] == "ok")
n_err = len(manifest) - n_ok
print(f"DONE total={len(manifest)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s")
