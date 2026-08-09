
import numpy as np, json, glob, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(PROJECT_ROOT / "data/cache_segmented_v3/chair__chair_range_10m__*_seg.npz")))
results = []
for f in files:
    d = np.load(f)
    xyz = d["xyz"]
    centroid = xyz[:, :3].mean(axis=0)
    bbox_min = xyz[:, :3].min(axis=0)
    bbox_max = xyz[:, :3].max(axis=0)
    results.append({"file": os.path.basename(f), "n": xyz.shape[0], "centroid": centroid.tolist(),
                     "bbox_min": bbox_min.tolist(), "bbox_max": bbox_max.tolist()})
print(json.dumps(results, indent=2))
