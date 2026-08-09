"""Render dummy-only PCN completion comparisons from the saved training checkpoints."""
from pathlib import Path

import sys

# The PCN architecture lives in the sibling 02_pcn_training folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_pcn_training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_from_dataset import PCN


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "Evaluation_Rendering" / "comparison_outputs"
DATASET = PROJECT_ROOT / "Route_B_Learned_Completion" / "01_simulation_dataset_generation" / "dataset_dummy.npz"
CHECKPOINTS = (
    (5, PROJECT_ROOT / "Route_B_Learned_Completion" / "02_pcn_training" / "pcn_dummy_5ep.pth"),
    (10, PROJECT_ROOT / "Route_B_Learned_Completion" / "02_pcn_training" / "pcn_dummy_10ep.pth"),
)
SAMPLE_INDEX = 42
MAX_POINTS = 8_000
RNG = np.random.default_rng(42)


def subsample(points: np.ndarray) -> np.ndarray:
    if len(points) <= MAX_POINTS:
        return points
    return points[RNG.choice(len(points), MAX_POINTS, replace=False)]


def predict(partial: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("classes") != ["dummy"]:
        raise ValueError(f"{checkpoint_path.name} is not a dummy-only checkpoint")
    model = PCN().eval()
    model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        _, completed = model(torch.from_numpy(partial).float().unsqueeze(0))
    return completed.squeeze(0).numpy()


def plot_points(ax, points: np.ndarray, title: str, color: str) -> None:
    points = subsample(points)
    center = points.mean(axis=0)
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.55, c=color, alpha=0.78, linewidths=0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold", pad=12)
    ax.set_axis_off()
    ax.set_facecolor("#0f172a")


def save_figure(panels, filename: str, title: str) -> None:
    figure = plt.figure(figsize=(5.2 * len(panels), 5.8), facecolor="#0f172a")
    figure.suptitle(title, color="white", fontsize=20, fontweight="bold", y=0.93)
    for index, (label, points, color) in enumerate(panels, start=1):
        axis = figure.add_subplot(1, len(panels), index, projection="3d")
        plot_points(axis, points, label, color)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.80, wspace=0.03)
    figure.savefig(OUT_DIR / filename, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    dataset = np.load(DATASET)
    partial = dataset["parts"][SAMPLE_INDEX]
    ground_truth = dataset["gts"][SAMPLE_INDEX]
    predictions = {}

    for epoch, path in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(path)
        prediction = predict(partial, path)
        predictions[epoch] = prediction
        save_figure(
            (("Input sonar partial", partial, "#38bdf8"),
             ("Synthetic ground truth", ground_truth, "#cbd5e1"),
             (f"PCN completion — epoch {epoch}", prediction, "#a78bfa")),
            f"dummy_epoch_{epoch:02d}_comparison.png",
            f"Dummy Point-Completion Output — Epoch {epoch}",
        )

    save_figure(
        (("Input sonar partial", partial, "#38bdf8"),
         ("Synthetic ground truth", ground_truth, "#cbd5e1"),
         ("PCN completion — epoch 5", predictions[5], "#a78bfa"),
         ("PCN completion — epoch 10", predictions[10], "#f97316")),
        "dummy_epoch_05_vs_10.png",
        "Dummy PCN Completion Progression: Epoch 5 vs Epoch 10",
    )
    print("saved dummy epoch comparison images to", OUT_DIR)


if __name__ == "__main__":
    main()
