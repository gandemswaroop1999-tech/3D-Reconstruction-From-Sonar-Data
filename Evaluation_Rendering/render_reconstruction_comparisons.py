"""Render chair and dummy reconstruction-method comparisons from saved project outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "comparison_outputs"
FUSED_DIR = PROJECT_ROOT / "data" / "cache_fused"
MESHES_DIR = PROJECT_ROOT / "Route_A_Geometric_Fusion" / "04_meshing" / "meshes"
OBJECTS = ("chair", "dummy")
METHODS = (
    ("Fused sonar cloud", "cloud", "#38bdf8"),
    ("Poisson reconstruction", "poisson", "#f97316"),
    ("Ball-pivoting reconstruction", "bpa", "#a78bfa"),
)
MAX_POINTS = 14_000
RNG = np.random.default_rng(42)


def sample_points(object_name: str, method: str) -> np.ndarray:
    if method == "cloud":
        points = np.load(FUSED_DIR / f"{object_name}_fused.npz")["xyz"]
    else:
        mesh = o3d.io.read_triangle_mesh(str(MESHES_DIR / f"{object_name}_{method}.ply"))
        points = np.asarray(mesh.sample_points_uniformly(number_of_points=MAX_POINTS).points)
    if len(points) > MAX_POINTS:
        points = points[RNG.choice(len(points), MAX_POINTS, replace=False)]
    return points


def plot_cloud(ax, points: np.ndarray, color: str, title: str) -> None:
    center = points.mean(axis=0)
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, c=color, alpha=0.72, linewidths=0)
    ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=21, azim=-58)
    ax.set_axis_off()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(16, 9), facecolor="#0f172a")
    fig.suptitle("Sonar 3D Reconstruction Comparison", color="white", fontsize=20, fontweight="bold", y=0.94)

    for row, object_name in enumerate(OBJECTS):
        for col, (label, method, color) in enumerate(METHODS):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            ax.set_facecolor("#0f172a")
            plot_cloud(ax, sample_points(object_name, method), color, f"{object_name.title()} — {label}")

    fig.text(0.5, 0.035, "Each mesh is displayed as uniformly sampled surface points for an equivalent visual comparison.",
             color="#cbd5e1", ha="center", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.08, wspace=0.02, hspace=0.13)
    output = OUT_DIR / "chair_dummy_reconstruction_comparison.png"
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    print(output)


if __name__ == "__main__":
    main()
