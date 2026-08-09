
import sys, os, json, time
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "cache_clean"
OUT_DIR = DATA_DIR / "cache_segmented_v3"
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_DIR / "case_paths_120.txt") as f:
    case_paths = [l.strip() for l in f if l.strip()]

NOMINAL_SIZES = {
    "tire": 0.66, "dummy": 1.4, "drum": 1.2, "chair": 0.4, "net": 0.5
}
PLANE_DIST_THRESH = 0.05
PLANE_ITERS = 200
MAX_PLANES = 3
MIN_PLANE_INLIER_FRAC = 0.15
EPS = 0.08
MIN_SAMPLES = 15
SIZE_TOLERANCE = 2.5
FLATNESS_RATIO_THRESH = 0.12

def ransac_plane(pts, dist_thresh, n_iters, seed=0):
    rng = np.random.default_rng(seed)
    n = pts.shape[0]
    best_inliers = None
    best_count = -1
    best_plane = None
    for _ in range(n_iters):
        idx3 = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx3]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len
        d = -normal.dot(p0)
        dist = np.abs(pts.dot(normal) + d)
        inliers = dist < dist_thresh
        cnt = inliers.sum()
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)
    return best_inliers, best_plane

def remove_dominant_planes(xyz, dist_thresh=PLANE_DIST_THRESH, n_iters=PLANE_ITERS,
                            max_planes=MAX_PLANES, min_inlier_frac=MIN_PLANE_INLIER_FRAC, seed=0):
    working = xyz.copy()
    planes_removed = 0
    for _ in range(max_planes):
        pts = working[:, :3]
        if pts.shape[0] < 50:
            break
        inliers, plane = ransac_plane(pts, dist_thresh=dist_thresh, n_iters=n_iters, seed=seed)
        frac = inliers.sum() / len(pts)
        if frac < min_inlier_frac:
            break
        working = working[~inliers]
        planes_removed += 1
    return working, planes_removed

def segment_object_v3(xyz, nominal_size_m, plane_dist_thresh=PLANE_DIST_THRESH, plane_iters=PLANE_ITERS,
                       max_planes=MAX_PLANES, eps=EPS, min_samples=MIN_SAMPLES, size_tolerance=SIZE_TOLERANCE,
                       flatness_ratio_thresh=FLATNESS_RATIO_THRESH, seed=0):
    working_xyz, n_planes = remove_dominant_planes(
        xyz, dist_thresh=plane_dist_thresh, n_iters=plane_iters,
        max_planes=max_planes, seed=seed
    )
    if working_xyz.shape[0] < min_samples:
        working_xyz = xyz
        n_planes = 0

    working_pts = working_xyz[:, :3]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(working_pts)
    labels = db.labels_
    unique = np.unique(labels)
    unique = unique[unique != -1]

    expected_diag = nominal_size_m * np.sqrt(3)
    candidates = []
    for u in unique:
        mask = labels == u
        cluster_pts = working_pts[mask]
        bbox = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        diag = np.linalg.norm(bbox)
        n = mask.sum()
        flatness = bbox.min() / max(bbox.max(), 1e-6)
        candidates.append({"label": u, "n": n, "bbox": bbox, "diag": diag, "mask": mask, "flatness": flatness})

    if not candidates:
        return None, {"status": "no_clusters", "n_planes_removed": n_planes}

    in_range = [c for c in candidates
                if (expected_diag/size_tolerance) <= c["diag"] <= (expected_diag*size_tolerance)
                and c["flatness"] >= flatness_ratio_thresh]

    if in_range:
        best = max(in_range, key=lambda c: c["n"])
        status = "size_matched"
    else:
        in_range_relaxed = [c for c in candidates if (expected_diag/size_tolerance) <= c["diag"] <= (expected_diag*size_tolerance)]
        if in_range_relaxed:
            best = max(in_range_relaxed, key=lambda c: c["n"])
            status = "size_matched_flat"
        else:
            plausible = [c for c in candidates if c["diag"] < expected_diag * 8 and c["flatness"] >= flatness_ratio_thresh]
            if plausible:
                best = max(plausible, key=lambda c: c["n"])
                status = "fallback_largest_non_background"
            else:
                best = max(candidates, key=lambda c: c["n"])
                status = "fallback_largest_overall"

    seg_xyz = working_xyz[best["mask"]]
    meta = {
        "status": status,
        "n_planes_removed": n_planes,
        "n_clusters_total": int(len(unique)),
        "chosen_n_points": int(best["n"]),
        "chosen_bbox": [float(x) for x in best["bbox"]],
        "chosen_diag": float(best["diag"]),
        "chosen_flatness": float(best["flatness"]),
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
        seg_xyz, meta = segment_object_v3(xyz, NOMINAL_SIZES[obj])
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

with open(os.path.join(OUT_DIR, "segment_manifest_v3.json"), "w") as f:
    json.dump(manifest, f, indent=2)

n_ok = sum(1 for m in manifest if m.get("status") in ("size_matched", "size_matched_flat", "fallback_largest_non_background", "fallback_largest_overall"))
n_err = len(manifest) - n_ok
print(f"DONE total={len(manifest)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s")
