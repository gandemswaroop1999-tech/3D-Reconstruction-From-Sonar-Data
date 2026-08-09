
import numpy as np
from scipy.spatial import cKDTree
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(PROJECT_ROOT / "data/cache_segmented_v3/chair__chair_range_10m__*_seg.npz")))
for f in files[:4]:
    d = np.load(f)
    xyz = d["xyz"][:, :3]
    if xyz.shape[0] < 5:
        continue
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=2)
    nn = dists[:, 1]
    print(f, "n=", xyz.shape[0], "nn_mean=%.4f nn_median=%.4f nn_p90=%.4f" % (nn.mean(), np.median(nn), np.percentile(nn, 90)))
