
import numpy as np
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_FUSED = PROJECT_ROOT / "data" / "cache_fused"

OBJECTS = ["tire", "dummy", "drum", "chair", "net"]
for obj in OBJECTS:
    d = np.load(CACHE_FUSED / f"{obj}_fused.npz")
    xyz = d["xyz"]
    n = xyz.shape[0]
    centroid = xyz.mean(axis=0)
    dists = np.linalg.norm(xyz - centroid, axis=1)
    p50 = np.percentile(dists, 50)
    p90 = np.percentile(dists, 90)
    p99 = np.percentile(dists, 99)
    dmax = dists.max()
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    # extent ignoring top/bottom 1% outliers
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    extent_robust = hi - lo
    print(obj, "n=", n, "dist_from_centroid p50=%.3f p90=%.3f p99=%.3f max=%.3f" % (p50, p90, p99, dmax),
          "full_extent=", np.round(extent, 3).tolist(), "robust_extent(1-99pct)=", np.round(extent_robust, 3).tolist())
