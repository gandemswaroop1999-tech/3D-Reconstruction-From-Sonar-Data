
import sys, os, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "cache_clean"
OUT_DIR = DATA_DIR / "cache_segmented"
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_DIR / "case_paths_120.txt") as f:
    case_paths = [l.strip() for l in f if l.strip()]

NOMINAL_SIZES = {
    "tire": 0.66, "dummy": 1.4, "drum": 1.2, "chair": 0.4, "net": 0.5
}
EPS = 0.08
MIN_SAMPLES = 15
SIZE_TOLERANCE = 2.5

def segment_object(xyz, nominal_size_m, eps=EPS, min_samples=MIN_SAMPLES, size_tolerance=SIZE_TOLERANCE):
    pts = xyz[:, :3]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique = np.unique(labels)
    unique = unique[unique != -1]

    expected_diag = nominal_size_m * np.sqrt(3)
    candidates = []
    for u in unique:
        mask = labels == u
        cluster_pts = pts[mask]
        bbox = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        diag = np.linalg.norm(bbox)
        n = mask.sum()
        candidates.append({"label": u, "n": n, "bbox": bbox, "diag": diag, "mask": mask})

    if not candidates:
        return None, {"status": "no_clusters"}

    in_range = [c for c in candidates if (expected_diag/size_tolerance) <= c["diag"] <= (expected_diag*size_tolerance)]

    if in_range:
        best = max(in_range, key=lambda c: c["n"])
        status = "size_matched"
    else:
        plausible = [c for c in candidates if c["diag"] < expected_diag * 8]
        if plausible:
            best = max(plausible, key=lambda c: c["n"])
            status = "fallback_largest_non_background"
        else:
            best = max(candidates, key=lambda c: c["n"])
            status = "fallback_largest_overall"

    seg_xyz = xyz[best["mask"]]
    meta = {
        "status": status,
        "n_clusters_total": int(len(unique)),
        "chosen_n_points": int(best["n"]),
        "chosen_bbox": [float(x) for x in best["bbox"]],
        "chosen_diag": float(best["diag"]),
        "expected_diag": float(expected_diag),
    }
    return seg_xyz, meta

manifest = []
t0 = time.time()
for i, cp in enumerate(case_paths):
    obj = cp.split("/")[0]
    in_name = cp.replace("/", "__") + "_clean.npz"
    out_name = cp.replace("/", "__") + "_seg.npz"
    in_path = os.path.join(CLEAN_DIR, in_name)
    out_path = os.path.join(OUT_DIR, out_name)
    try:
        d = np.load(in_path)
        xyz = d["xyz"]
        seg_xyz, meta = segment_object(xyz, NOMINAL_SIZES[obj])
        if seg_xyz is None:
            meta["case_path"] = cp
            meta["object"] = obj
            manifest.append(meta)
            continue
        np.savez_compressed(out_path, xyz=seg_xyz.astype(np.float32))
        meta["case_path"] = cp
        meta["object"] = obj
        meta["n_input"] = int(xyz.shape[0])
    except Exception as e:
        meta = {"case_path": cp, "object": obj, "status": "error", "error": str(e)}
    manifest.append(meta)
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{len(case_paths)}] elapsed={time.time()-t0:.1f}s", flush=True)

with open(os.path.join(OUT_DIR, "segment_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

n_ok = sum(1 for m in manifest if m.get("status") in ("size_matched", "fallback_largest_non_background", "fallback_largest_overall"))
n_err = len(manifest) - n_ok
print(f"DONE total={len(manifest)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s")
