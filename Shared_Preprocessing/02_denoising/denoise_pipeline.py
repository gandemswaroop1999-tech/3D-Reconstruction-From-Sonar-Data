
import sys, os, json, time
from pathlib import Path
import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUT_DIR = DATA_DIR / "cache_clean"
os.makedirs(OUT_DIR, exist_ok=True)

with open(DATA_DIR / "case_paths_120.txt") as f:
    case_paths = [l.strip() for l in f if l.strip()]

# Preprocessing parameters
SOR_NB_NEIGHBORS = 16
SOR_STD_RATIO = 2.0
VOXEL_SIZE = 0.01  # 1cm downsample voxel

manifest = []
t0 = time.time()
for i, cp in enumerate(case_paths):
    in_name = cp.replace("/", "__") + ".npz"
    out_name = cp.replace("/", "__") + "_clean.npz"
    in_path = os.path.join(CACHE_DIR, in_name)
    out_path = os.path.join(OUT_DIR, out_name)
    try:
        d = np.load(in_path)
        xyz = d["xyz"]  # (N,4) float32: X,Y,Z,Intensity
        n_raw = xyz.shape[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3].astype(np.float64))

        # Statistical Outlier Removal
        pcd_sor, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO
        )
        intensity_sor = xyz[inlier_idx, 3]
        n_after_sor = len(inlier_idx)

        # Voxel downsample (need to track intensity via a workaround: use point index to voxel via numpy)
        pts_sor = np.asarray(pcd_sor.points)
        voxel_idx = np.floor(pts_sor / VOXEL_SIZE).astype(np.int64)
        _, unique_idx, inverse = np.unique(voxel_idx, axis=0, return_index=True, return_inverse=True)
        # Average points and intensity within each voxel
        n_voxels = unique_idx.shape[0]
        sums_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
        sums_i = np.zeros((n_voxels,), dtype=np.float64)
        counts = np.zeros((n_voxels,), dtype=np.int64)
        np.add.at(sums_xyz, inverse, pts_sor)
        np.add.at(sums_i, inverse, intensity_sor)
        np.add.at(counts, inverse, 1)
        avg_xyz = sums_xyz / counts[:, None]
        avg_i = sums_i / counts
        n_final = n_voxels

        out_xyz = np.column_stack([avg_xyz, avg_i]).astype(np.float32)
        np.savez_compressed(out_path, xyz=out_xyz, n_points=n_final)

        meta = {
            "case_path": cp,
            "n_raw": int(n_raw),
            "n_after_sor": int(n_after_sor),
            "n_final_downsampled": int(n_final),
            "sor_removed_frac": float(1 - n_after_sor / n_raw),
            "downsample_removed_frac": float(1 - n_final / n_after_sor) if n_after_sor > 0 else None,
            "status": "ok",
        }
    except Exception as e:
        meta = {"case_path": cp, "status": "error", "error": str(e)}
    manifest.append(meta)
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/{len(case_paths)}] elapsed={time.time()-t0:.1f}s", flush=True)

with open(os.path.join(OUT_DIR, "clean_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

n_ok = sum(1 for m in manifest if m["status"] == "ok")
n_err = len(manifest) - n_ok
print(f"DONE total={len(manifest)} ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s")
