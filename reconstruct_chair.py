"""Measurement-aware SUOP folding-chair reconstruction.

The SUOP scenes contain a moved target inside a much larger stationary seabed
scene.  This script range-gates the high-SNR 3 m chair cases, removes the seabed,
subtracts geometry repeated across acquisitions, and extracts chair-scale moving
clusters.  The retained observations are registered independently to a
dimensionally correct folding-chair prior, which completes surfaces and frame
members that the sonar did not observe.

Only chair data are read or written by the production path in this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cache_clean"
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "suop_detection_reference"
    / "IOES-Lab-SUOP-Object-Detection-0441a44"
    / "object_detection"
    / "YOLOv8"
    / "bbox_labels"
    / "chair"
)
OUTPUT_DIR = ROOT / "chair_outputs"

# Exact view used by the official SUOP benchmark's png_make.py.
ROT_X_DEG = -5.0
ROT_Y_DEG = -170.0
ROT_Z_DEG = 180.0


def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


BENCHMARK_ROTATION = (
    rotation_matrix("z", np.deg2rad(ROT_Z_DEG))
    @ rotation_matrix("y", np.deg2rad(ROT_Y_DEG))
    @ rotation_matrix("x", np.deg2rad(ROT_X_DEG))
)


def parse_case_from_cache(path: Path) -> tuple[str, str]:
    # chair__chair_range_3m__case_060_clean.npz
    tokens = path.stem.split("__")
    return tokens[1].replace("chair_range_", ""), tokens[2].replace("_clean", "")


def read_official_bbox(range_name: str, case_name: str) -> np.ndarray:
    label = REFERENCE_ROOT / f"chair_range_{range_name}" / f"{case_name}.txt"
    if not label.exists():
        raise FileNotFoundError(
            f"Official SUOP label missing: {label}. Extract the SUOP Object "
            "Detection v1.1 reference under suop_detection_reference/."
        )
    values = np.fromstring(label.read_text(encoding="utf-8"), sep=" ")
    if values.size != 5 or int(values[0]) != 3:
        raise ValueError(f"Unexpected chair label in {label}: {values}")
    bbox = values[1:].copy()  # normalized x-center, y-center, width, height
    # The released labels use the horizontally mirrored convention of the
    # benchmark training images, while png_make.py/Matplotlib emits the view in
    # display coordinates.  Undo that convention when lifting back to XYZ.
    bbox[0] = 1.0 - bbox[0]
    return bbox


def load_original_cloud(range_name: str, case_name: str) -> np.ndarray:
    path = (
        PROJECT_ROOT / "suop_repo" / "SUOP_dataset" / "chair"
        / f"chair_range_{range_name}" / case_name / "point_cloud.xyz"
    )
    if not path.exists() or path.stat().st_size < 1000:
        raise FileNotFoundError(f"Original SUOP cloud is unavailable (or still an LFS pointer): {path}")
    xyz = np.loadtxt(path, dtype=np.float64)
    return xyz.reshape(-1, xyz.shape[-1])


def benchmark_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return benchmark-rotated XYZ and normalized screen coordinates.

    Normalization is relative to the square 3-D axes.  This is also the crop
    produced by bbox_inches='tight', so it maps directly to the official YOLO
    coordinates (top-left image origin).
    """
    rotated = points @ BENCHMARK_ROTATION.T
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    # Set the same automatic limits/margins as Axes3D.scatter without actually
    # rasterizing ~150k points merely to obtain their projection matrix.
    ax.auto_scale_xyz(rotated[:, 0], rotated[:, 1], rotated[:, 2])
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()
    fig.tight_layout()
    fig.canvas.draw()
    xp, yp, _ = proj3d.proj_transform(
        rotated[:, 0], rotated[:, 1], rotated[:, 2], ax.get_proj()
    )
    display = ax.transData.transform(np.column_stack((xp, yp)))
    u = (display[:, 0] - ax.bbox.x0) / ax.bbox.width
    v = 1.0 - (display[:, 1] - ax.bbox.y0) / ax.bbox.height
    plt.close(fig)
    return rotated, np.column_stack((u, v))


def robust_floor_height(points: np.ndarray, nominal_range: float) -> float:
    radius = np.linalg.norm(points[:, :2], axis=1)
    shell = points[(radius > nominal_range - 1.0) & (radius < nominal_range + 1.0)]
    if len(shell) < 100:
        shell = points
    lo, hi = np.quantile(shell[:, 2], [0.02, 0.98])
    edges = np.arange(lo, hi + 0.01, 0.01)
    hist, edges = np.histogram(shell[:, 2], bins=edges)
    peak = int(hist.argmax())
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def segment_case(path: Path, bbox_padding: float = 0.16) -> dict:
    range_name, case_name = parse_case_from_cache(path)
    nominal_range = float(range_name.removesuffix("m"))
    # The official boxes were annotated on renders of the original XYZ files.
    # Denoising changes Matplotlib's automatic 3-D limits and therefore moves
    # points in screen space, so projection must happen before filtering.
    xyz = load_original_cloud(range_name, case_name)
    points = xyz[:, :3].astype(np.float64)
    intensity = xyz[:, 3].astype(np.float64) if xyz.shape[1] > 3 else np.ones(len(points))
    _, screen = benchmark_projection(points)
    cx, cy, width, height = read_official_bbox(range_name, case_name)
    half_w = width * (0.5 + bbox_padding)
    half_h = height * (0.5 + bbox_padding)
    in_box = (
        (np.abs(screen[:, 0] - cx) <= half_w)
        & (np.abs(screen[:, 1] - cy) <= half_h)
    )

    acoustic_range = np.linalg.norm(points, axis=1)
    floor_z = robust_floor_height(points, nominal_range)
    # The vertically mounted BV5000 uses +Z toward the seabed.  The published
    # chair is 0.4 m high; retain a little tolerance above and at its contact.
    height_mask = (points[:, 2] >= floor_z - 0.52) & (points[:, 2] <= floor_z + 0.025)
    # SUOP's range is slant range from the sonar, not horizontal XY range.
    range_mask = (
        (acoustic_range >= nominal_range - 0.45)
        & (acoustic_range <= nominal_range + 0.45)
    )
    keep = in_box & height_mask & range_mask
    proposal = points[keep]
    proposal_intensity = intensity[keep]
    if len(proposal) < 30:
        raise RuntimeError(f"{range_name}/{case_name}: only {len(proposal)} proposal points")

    # A local frustum can still include isolated multipath returns.  Radius
    # outlier removal keeps separate chair parts, unlike choosing one DBSCAN
    # component (which would often discard the seat or a leg).
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proposal))
    if len(proposal) >= 80:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)
        proposal = proposal[np.asarray(ind)]
        proposal_intensity = proposal_intensity[np.asarray(ind)]

    # Remove any residual seabed sheet while keeping chair-to-ground contacts.
    proposal_keep = proposal[:, 2] < floor_z - 0.012
    if proposal_keep.sum() >= 30:
        proposal = proposal[proposal_keep]
        proposal_intensity = proposal_intensity[proposal_keep]

    return {
        "range": range_name,
        "case": case_name,
        "points": proposal,
        "intensity": proposal_intensity,
        "screen": screen,
        "bbox": np.array([cx, cy, width, height]),
        "floor_z": floor_z,
        "source_count": len(points),
        "frustum_count": int(in_box.sum()),
    }


def voxel_down(points: np.ndarray, voxel: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(pcd.voxel_down_sample(voxel).points)


def prepare_temporal_scan(path: Path) -> dict:
    """Range-gate one scene and remove its dominant seabed plane."""
    range_name, case_name = parse_case_from_cache(path)
    nominal_range = float(range_name.removesuffix("m"))
    xyz = load_original_cloud(range_name, case_name)
    points = xyz[:, :3]
    intensity = xyz[:, 3] if xyz.shape[1] > 3 else np.ones(len(points))
    slant_range = np.linalg.norm(points, axis=1)
    shell_mask = np.abs(slant_range - nominal_range) <= 0.52
    shell = points[shell_mask]
    shell_intensity = intensity[shell_mask]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shell))
    plane, _ = pcd.segment_plane(distance_threshold=0.020, ransac_n=3, num_iterations=1800)
    normal = np.asarray(plane[:3])
    plane_distance = np.abs(shell @ normal + plane[3])
    off_plane = plane_distance > 0.035
    shell = shell[off_plane]
    shell_intensity = shell_intensity[off_plane]

    # Trace original samples through voxel selection by retaining the strongest
    # return per 12 mm cell.  Strongest-return selection is preferable to a mean
    # for sonar because it avoids averaging multipath points into a false surface.
    cells = np.floor(shell / 0.012).astype(np.int64)
    order = np.argsort(shell_intensity)[::-1]
    _, first = np.unique(cells[order], axis=0, return_index=True)
    chosen = order[first]
    return {
        "range": range_name,
        "case": case_name,
        "points": shell[chosen],
        "intensity": shell_intensity[chosen],
        "plane": [float(v) for v in plane],
        "source_count": len(points),
    }


def temporal_chair_segments(files: list[Path]) -> tuple[list[dict], list[dict]]:
    """Subtract persistent scene geometry and select chair-sized transients."""
    o3d.utility.random.seed(7)
    scans = [prepare_temporal_scan(path) for path in files]
    trees = [cKDTree(scan["points"]) for scan in scans]
    segments: list[dict] = []
    diagnostics: list[dict] = []
    for scan_index, scan in enumerate(scans):
        points = scan["points"]
        support = np.zeros(len(points), dtype=np.int16)
        for other_index, tree in enumerate(trees):
            if other_index == scan_index:
                continue
            distance, _ = tree.query(points, k=1, workers=-1)
            support += distance < 0.025
        # Geometry present in at least three *other* cases is stationary scene.
        transient_mask = support <= 2
        transient = points[transient_mask]
        transient_intensity = scan["intensity"][transient_mask]
        labels = DBSCAN(eps=0.032, min_samples=5, n_jobs=-1).fit_predict(transient)

        candidates: list[dict] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            cluster = transient[mask]
            if len(cluster) < 20:
                continue
            span = np.ptp(cluster, axis=0)
            diagonal = float(np.linalg.norm(span))
            center = np.median(cluster, axis=0)
            center_range = float(np.linalg.norm(center))
            if not (0.16 <= diagonal <= 0.72 and 2.55 <= center_range <= 3.48):
                continue
            # The paper specifies a 0.2 x 0.4 m target.  Reward chair-scale
            # support and penalize range/size deviations without forcing an AABB
            # orientation (the chair is deliberately rotated between cases).
            size_weight = math.exp(-0.5 * ((diagonal - 0.45) / 0.20) ** 2)
            range_weight = math.exp(-0.5 * ((center_range - 3.0) / 0.38) ** 2)
            score = len(cluster) * (0.35 + 0.65 * size_weight) * range_weight
            candidates.append({
                "label": int(label), "mask": mask, "n": len(cluster),
                "span": span, "diagonal": diagonal, "center": center,
                "center_range": center_range, "score": float(score),
            })
        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        if not candidates:
            diagnostics.append({
                "case": scan["case"], "status": "no_chair_scale_transient",
                "n_transient": len(transient), "plane": scan["plane"],
            })
            continue
        best = candidates[0]
        cluster_mask = best["mask"]
        segment = {
            "range": scan["range"], "case": scan["case"],
            "points": transient[cluster_mask],
            "intensity": transient_intensity[cluster_mask],
            "score": best["score"], "diagonal": best["diagonal"],
            "center_range": best["center_range"], "span": best["span"],
            "candidate_count": len(candidates), "plane": scan["plane"],
        }
        segments.append(segment)
        diagnostics.append({
            "case": scan["case"], "status": "selected", "n_shell": len(points),
            "n_transient": len(transient), "n_points": best["n"],
            "score": best["score"], "diagonal": best["diagonal"],
            "center_range": best["center_range"], "span": best["span"].tolist(),
            "candidate_count": len(candidates), "plane": scan["plane"],
        })
    return segments, diagnostics


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float, color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    vector = end - start
    length = float(np.linalg.norm(vector))
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=20, split=2)
    direction = vector / length
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    if cross_norm > 1e-9:
        angle = math.acos(float(np.clip(z_axis @ direction, -1.0, 1.0)))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(cross / cross_norm * angle), center=(0, 0, 0))
    elif direction[2] < 0:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle((math.pi, 0, 0)), center=(0, 0, 0))
    mesh.translate((start + end) / 2.0)
    mesh.paint_uniform_color(color)
    return mesh


def box_centered(size: tuple[float, float, float], center: tuple[float, float, float], color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(np.asarray(center) - np.asarray(size) / 2.0)
    mesh.paint_uniform_color(color)
    return mesh


def make_folding_chair_mesh() -> o3d.geometry.TriangleMesh:
    """Dimensionally faithful 0.2 m x 0.4 m SUOP folding-chair prior."""
    plastic = (0.72, 0.78, 0.84)
    steel = (0.18, 0.24, 0.31)
    mesh = box_centered((0.22, 0.19, 0.016), (0.0, -0.005, 0.215), plastic)
    back = box_centered((0.22, 0.018, 0.115), (0.0, 0.087, 0.338), plastic)
    # A small rearward back-panel rake matches the photographed SUOP target.
    back.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle((-math.radians(7), 0, 0)), center=(0, 0.087, 0.28))
    mesh += back
    radius = 0.006
    for x in (-0.103, 0.103):
        # Two crossed folding-frame members on each side.
        mesh += cylinder_between(np.array([x, -0.125, 0.0]), np.array([x, 0.075, 0.245]), radius, steel)
        mesh += cylinder_between(np.array([x, 0.125, 0.0]), np.array([x, -0.075, 0.245]), radius, steel)
        mesh += cylinder_between(np.array([x, 0.070, 0.205]), np.array([x, 0.105, 0.405]), radius, steel)
    # Cross-members materially visible to a 1.35 MHz sonar.
    for y, z in ((-0.095, 0.205), (0.090, 0.205), (0.100, 0.302)):
        mesh += cylinder_between(np.array([-0.103, y, z]), np.array([0.103, y, z]), radius, steel)
    mesh.compute_vertex_normals()
    return mesh


def signed_permutation_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return matrices


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    return center, vectors[:, np.argsort(values)[::-1]]


def fit_segment_to_prior(points: np.ndarray, model_points: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a moved/rotated observation to the canonical chair model."""
    source = voxel_down(points, 0.006)
    target = voxel_down(model_points, 0.004)
    source_center, source_axes = principal_axes(source)
    target_center, target_axes = principal_axes(target)
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=40))
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))

    best_result = None
    best_quality = (-1.0, float("inf"))
    for signed_permutation in signed_permutation_matrices():
        rotation = target_axes @ signed_permutation @ source_axes.T
        initial = np.eye(4)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_center - rotation @ source_center
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.055, initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=0.028)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=70),
        )
        quality = (float(result.fitness), float(result.inlier_rmse))
        if quality[0] > best_quality[0] + 1e-6 or (
            abs(quality[0] - best_quality[0]) <= 1e-6 and quality[1] < best_quality[1]
        ):
            best_quality = quality
            best_result = result
    assert best_result is not None
    transformed = (
        points @ best_result.transformation[:3, :3].T
        + best_result.transformation[:3, 3]
    )
    return transformed, {
        "fitness": float(best_result.fitness),
        "rmse": float(best_result.inlier_rmse),
        "transform_to_canonical": best_result.transformation.tolist(),
    }


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def trimmed_nn_error(source: np.ndarray, target_tree: cKDTree, trim: float = 0.72) -> float:
    distances, _ = target_tree.query(source, k=1, workers=-1)
    count = max(8, int(len(distances) * trim))
    return float(np.mean(np.partition(distances, count - 1)[:count]))


def register_yaw(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Register an upright chair scan using yaw search plus robust ICP.

    SUOP changes chair position/yaw between cases while the sonar remains fixed.
    Constraining the global search to the physical acquisition degrees of
    freedom is substantially more stable than unconstrained FPFH on sparse legs.
    """
    voxel = 0.006
    src = voxel_down(source, voxel)
    tgt = voxel_down(target, voxel)
    src_center = np.median(src, axis=0)
    tgt_center = np.median(tgt, axis=0)
    src0, tgt0 = src - src_center, tgt - tgt_center
    tree = cKDTree(tgt0)

    candidates: list[tuple[float, float]] = []
    for degrees in np.arange(0.0, 360.0, 5.0):
        angle = np.deg2rad(degrees)
        error = trimmed_nn_error(src0 @ yaw_matrix(angle).T, tree)
        candidates.append((error, angle))
    _, best_angle = min(candidates)
    for step_deg in (1.0, 0.25):
        angles = best_angle + np.deg2rad(np.arange(-5, 6) * step_deg)
        scored = [(trimmed_nn_error(src0 @ yaw_matrix(a).T, tree), a) for a in angles]
        _, best_angle = min(scored)

    rot = yaw_matrix(best_angle)
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = tgt_center - rot @ src_center

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=40))
    tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=40))
    result = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        0.025,
        transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.012)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    registered = (source @ result.transformation[:3, :3].T) + result.transformation[:3, 3]
    metrics = {
        "yaw_deg": float(np.rad2deg(best_angle) % 360.0),
        "fitness": float(result.fitness),
        "rmse": float(result.inlier_rmse),
        "transform": result.transformation.tolist(),
    }
    return registered, metrics


def render_segmentation_grid(segments: list[dict], output: Path) -> None:
    fig = plt.figure(figsize=(16, 8.5), facecolor="#08111f")
    fig.suptitle(
        "SUOP Chair — temporal-background-subtracted 3D proposals (3 m scans)",
        color="white",
        fontsize=20,
        fontweight="bold",
        y=0.96,
    )
    for index, segment in enumerate(segments, start=1):
        ax = fig.add_subplot(2, 4, index, projection="3d")
        points = segment["points"]
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="turbo",
            s=1.2, linewidths=0, alpha=0.9,
        )
        center = points.mean(axis=0)
        radius = max(np.ptp(points, axis=0).max() / 2.0, 0.05)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-58)
        ax.set_axis_off()
        ax.set_facecolor("#08111f")
        ax.set_title(
            f"{segment['case']}  •  {len(points):,} pts",
            color="#dbeafe", fontsize=11, fontweight="bold",
        )
    fig.text(
        0.5, 0.025,
        "Persistent seabed/tripod geometry is removed; each panel is the best chair-scale moving cluster.",
        color="#8fa6bf", ha="center", fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.90, wspace=0.02, hspace=0.13)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_label_debug(path: Path, output: Path) -> None:
    """Render the official benchmark view with its published chair box."""
    range_name, case_name = parse_case_from_cache(path)
    points = load_original_cloud(range_name, case_name)[:, :3]
    rotated = points @ BENCHMARK_ROTATION.T
    z = rotated[:, 2]
    norm = (z - z.min()) / (np.ptp(z) + 1e-8)
    colors = plt.get_cmap("jet")(np.clip((norm - 0.3) / 0.8, 0.0, 1.0))[:, :3]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        rotated[:, 0], rotated[:, 1], rotated[:, 2], c=colors, s=0.1,
        depthshade=False, linewidths=0,
    )
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0, facecolor="#050914")
    plt.close(fig)
    # YOLO annotations are normalized to the final tight-cropped image.
    cx, cy, width, height = read_official_bbox(range_name, case_name)
    image = Image.open(output).convert("RGB")
    iw, ih = image.size
    box = (
        int((cx - width / 2) * iw), int((cy - height / 2) * ih),
        int((cx + width / 2) * iw), int((cy + height / 2) * ih),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(255, 255, 255), width=max(2, iw // 400))
    draw.text((box[0], max(0, box[1] - 18)), f"{range_name}/{case_name} chair", fill=(255, 255, 255))
    image.save(output)


def configure_3d_axis(ax, points: np.ndarray, elev: float, azim: float) -> None:
    center = np.mean(points, axis=0)
    radius = max(np.ptp(points, axis=0).max() / 2.0, 0.06) * 1.08
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor("#08111f")


def render_reconstruction(
    model_points: np.ndarray,
    fused_observed: np.ndarray,
    best_raw: np.ndarray,
    accepted_cases: list[str],
    metrics: dict,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor="#08111f")
    fig.suptitle(
        "SUOP Folding Chair — measurement-aware 3D reconstruction",
        color="white", fontsize=22, fontweight="bold", y=0.96,
    )
    panels = (
        (best_raw, None, "Best background-subtracted sonar observation", "#38bdf8", 20, -55),
        (fused_observed, None, f"Registered sonar evidence ({len(accepted_cases)} poses)", "#22d3ee", 22, -55),
        (model_points, None, "Completed folding-chair surface", "#d7e0ea", 20, -55),
        (model_points, fused_observed, "Evidence overlay: sonar on reconstructed surface", "#94a3b8", 20, -55),
    )
    for index, (base, overlay, title, color, elev, azim) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        if len(base) > 14000:
            selection = np.random.default_rng(7).choice(len(base), 14000, replace=False)
            shown = base[selection]
        else:
            shown = base
        ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=1.2, c=color, alpha=0.82, linewidths=0)
        extent_points = shown
        if overlay is not None and len(overlay):
            ax.scatter(overlay[:, 0], overlay[:, 1], overlay[:, 2], s=7.0, c="#f97316", alpha=0.9, linewidths=0)
            extent_points = np.vstack((shown, overlay))
        configure_3d_axis(ax, extent_points, elev, azim)
        ax.set_title(title, color="#e5edf7", fontsize=14, fontweight="bold", pad=12)
    fig.text(
        0.5, 0.025,
        f"Accepted cases: {', '.join(accepted_cases)}  •  observed→surface median {metrics['observed_to_surface_median_mm']:.1f} mm  "
        f"•  95th percentile {metrics['observed_to_surface_p95_mm']:.1f} mm",
        color="#9fb2c8", ha="center", fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.90, wspace=0.02, hspace=0.12)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_turntable(model_points: np.ndarray, fused_observed: np.ndarray, output: Path) -> None:
    fig = plt.figure(figsize=(16, 4.5), facecolor="#08111f")
    for index, azimuth in enumerate((-45, 45, 135, 225), start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        shown = model_points[::2]
        ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=0.8, c="#cbd5e1", alpha=0.75, linewidths=0)
        ax.scatter(fused_observed[:, 0], fused_observed[:, 1], fused_observed[:, 2], s=4.0, c="#f97316", alpha=0.85, linewidths=0)
        configure_3d_axis(ax, model_points, 18, azimuth)
        ax.set_title(f"Azimuth {azimuth % 360}°", color="#dbeafe", fontsize=11, fontweight="bold")
    fig.suptitle("Completed chair (grey) with registered sonar evidence (orange)", color="white", fontsize=17, fontweight="bold", y=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("segment", "all"), default="segment")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(CLEAN_DIR.glob("chair__chair_range_3m__case_*_clean.npz"))
    if not files:
        raise FileNotFoundError("No cached 3 m chair scans found")
    segments, manifest = temporal_chair_segments(files)
    if not segments:
        raise RuntimeError("Temporal subtraction found no chair-scale moving clusters")
    for segment in segments:
        np.savez_compressed(
            args.output_dir / f"{segment['case']}_chair_temporal_segment.npz",
            xyz=segment["points"].astype(np.float32),
            intensity=segment["intensity"].astype(np.float32),
        )
        print(
            f"{segment['case']}: chair={len(segment['points']):,} "
            f"diag={segment['diagonal']:.3f}m range={segment['center_range']:.3f}m "
            f"score={segment['score']:.1f}", flush=True,
        )
    render_segmentation_grid(segments, args.output_dir / "chair_temporal_segments.png")
    (args.output_dir / "segmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.stage == "segment":
        print(args.output_dir / "chair_temporal_segments.png")
        return

    o3d.utility.random.seed(11)
    chair_mesh = make_folding_chair_mesh()
    model_points = np.asarray(chair_mesh.sample_points_uniformly(number_of_points=30000).points)
    fitted: list[dict] = []
    for segment in segments:
        transformed, fit = fit_segment_to_prior(segment["points"], model_points)
        fit.update({"case": segment["case"], "n_points": len(segment["points"]), "score": segment["score"]})
        fitted.append({"points": transformed, "metrics": fit, "segment": segment})
        print(f"fit {segment['case']}: fitness={fit['fitness']:.3f} rmse={fit['rmse']*1000:.1f}mm", flush=True)

    # Require geometrically consistent support.  Always retain the best fit so
    # a very sparse dataset still produces an auditable single-view result.
    accepted = [
        item for item in fitted
        if item["metrics"]["n_points"] >= 60
        and item["metrics"]["fitness"] >= 0.42
        and item["metrics"]["rmse"] <= 0.038
    ]
    if not accepted:
        accepted = [max(fitted, key=lambda item: (item["metrics"]["fitness"], -item["metrics"]["rmse"]))]
    fused = voxel_down(np.vstack([item["points"] for item in accepted]), 0.004)
    model_tree = cKDTree(model_points)
    observed_distances, _ = model_tree.query(fused, k=1, workers=-1)
    observed_tree = cKDTree(fused)
    surface_distances, _ = observed_tree.query(model_points, k=1, workers=-1)
    accepted_cases = [item["metrics"]["case"] for item in accepted]
    best_item = max(accepted, key=lambda item: (item["metrics"]["fitness"], -item["metrics"]["rmse"]))
    metrics = {
        "method": "temporal background subtraction + robust model-assisted multi-view registration",
        "chair_prior_dimensions_m": {"length": 0.20, "height": 0.40, "seat_width": 0.22},
        "accepted_cases": accepted_cases,
        "observed_points_fused": len(fused),
        "observed_to_surface_median_mm": float(np.median(observed_distances) * 1000),
        "observed_to_surface_p95_mm": float(np.quantile(observed_distances, 0.95) * 1000),
        "surface_coverage_within_20mm": float(np.mean(surface_distances <= 0.020)),
        "fits": [item["metrics"] for item in fitted],
    }
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused))
    pcd.paint_uniform_color((0.10, 0.75, 0.88))
    o3d.io.write_point_cloud(str(args.output_dir / "chair_observed_fused.ply"), pcd)
    o3d.io.write_triangle_mesh(str(args.output_dir / "chair_reconstruction.ply"), chair_mesh, write_vertex_colors=True)
    (args.output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_reconstruction(
        model_points, fused, best_item["segment"]["points"], accepted_cases, metrics,
        args.output_dir / "chair_reconstruction.png",
    )
    render_turntable(model_points, fused, args.output_dir / "chair_reconstruction_turntable.png")
    print(args.output_dir / "chair_reconstruction.png")


if __name__ == "__main__":
    main()
