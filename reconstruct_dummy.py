"""Measurement-aware SUOP dummy reconstruction.

The SUOP dummy is moved between acquisitions while the seabed and test rig stay
fixed.  This script uses the high-SNR 3 m scans, removes the dominant seabed,
subtracts persistent scene geometry, extracts human-scale transient clusters,
and independently registers them to a dimensionally constrained 1.4 m dummy
prior.  The output keeps measured sonar evidence separate from prior-completed
surface geometry so the result is inspectable rather than a hallucinated mesh.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label as connected_labels
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from reconstruct_chair import (
    benchmark_projection,
    box_centered,
    configure_3d_axis,
    principal_axes,
    signed_permutation_matrices,
    voxel_down,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "cache_clean"
OUTPUT_DIR = ROOT / "dummy_outputs"
LABEL_ROOT = (
    PROJECT_ROOT / "suop_detection_reference" / "IOES-Lab-SUOP-Object-Detection-0441a44"
    / "object_detection" / "YOLOv8" / "bbox_labels" / "dummy" / "dummy_range_3m"
)


def parse_case(path: Path) -> str:
    return path.stem.split("__")[2].replace("_clean", "")


def load_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        xyz = np.asarray(archive["xyz"], dtype=np.float64)
    points = xyz[:, :3]
    intensity = xyz[:, 3] if xyz.shape[1] > 3 else np.ones(len(points))
    finite = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
    return points[finite], intensity[finite]


def load_original_dummy(case: str) -> tuple[np.ndarray, np.ndarray]:
    path = PROJECT_ROOT / "suop_repo" / "SUOP_dataset" / "dummy" / "dummy_range_3m" / case / "point_cloud.xyz"
    if not path.exists() or path.stat().st_size < 1000:
        raise FileNotFoundError(f"Original dummy cloud is unavailable: {path}")
    xyz = np.loadtxt(path, dtype=np.float64)
    points = xyz[:, :3]
    intensity = xyz[:, 3] if xyz.shape[1] > 3 else np.ones(len(points))
    return points, intensity


def read_dummy_bbox(case: str) -> np.ndarray:
    values = np.fromstring((LABEL_ROOT / f"{case}.txt").read_text(encoding="utf-8"), sep=" ")
    if values.size != 5 or int(values[0]) != 1:
        raise ValueError(f"Unexpected official dummy label for {case}: {values}")
    bbox = values[1:].copy()
    return bbox


def official_dummy_segments(files: list[Path]) -> tuple[list[dict], list[dict]]:
    """Lift the dataset's official 2-D dummy boxes back to original sonar XYZ."""
    o3d.utility.random.seed(29)
    segments: list[dict] = []
    diagnostics: list[dict] = []
    for cache_path in files:
        case = parse_case(cache_path)
        points, intensity = load_original_dummy(case)
        _, screen = benchmark_projection(points)
        cx, cy, width, height = read_dummy_bbox(case)
        pad = 0.12
        in_box = (
            (np.abs(screen[:, 0] - cx) <= width * (0.5 + pad))
            & (np.abs(screen[:, 1] - cy) <= height * (0.5 + pad))
        )
        slant = np.linalg.norm(points, axis=1)
        proposal_mask = in_box & (slant >= 1.85) & (slant <= 3.85)
        proposal = points[proposal_mask]
        proposal_intensity = intensity[proposal_mask]

        shell = points[(slant >= 1.85) & (slant <= 3.85)]
        shell_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shell))
        plane, _ = shell_pcd.segment_plane(distance_threshold=0.018, ransac_n=3, num_iterations=2200)
        normal = np.asarray(plane[:3])
        above_floor = np.abs(proposal @ normal + plane[3]) > 0.030
        proposal, proposal_intensity = proposal[above_floor], proposal_intensity[above_floor]
        if len(proposal) >= 80:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(proposal))
            _, kept = pcd.remove_statistical_outlier(nb_neighbors=14, std_ratio=1.8)
            kept = np.asarray(kept)
            proposal, proposal_intensity = proposal[kept], proposal_intensity[kept]
        proposal, proposal_intensity = strongest_per_voxel(proposal, proposal_intensity, 0.009)

        labels = DBSCAN(eps=0.048, min_samples=6, n_jobs=-1).fit_predict(proposal)
        candidates: list[tuple[float, np.ndarray, float]] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            member = labels == label
            cluster = proposal[member]
            if len(cluster) < 45:
                continue
            cluster_diagonal = float(np.linalg.norm(np.ptp(cluster, axis=0)))
            if not 0.50 <= cluster_diagonal <= 1.75:
                continue
            size_weight = math.exp(-0.5 * ((cluster_diagonal - 1.35) / 0.38) ** 2)
            candidates.append((len(cluster) * (0.25 + 0.75 * size_weight), member, cluster_diagonal))
        if candidates:
            _, member, _ = max(candidates, key=lambda candidate: candidate[0])
            proposal, proposal_intensity = proposal[member], proposal_intensity[member]

        if len(proposal) < 40:
            diagnostics.append({"case": case, "status": "official_box_too_sparse", "n_points": len(proposal)})
            continue
        span = np.ptp(proposal, axis=0)
        diagonal = float(np.linalg.norm(span))
        segment = {
            "case": case,
            "points": proposal,
            "intensity": proposal_intensity,
            "score": float(len(proposal)),
            "diagonal": diagonal,
            "center_range": float(np.linalg.norm(np.median(proposal, axis=0))),
            "span": span,
            "candidate_count": 1,
            "plane": [float(value) for value in plane],
        }
        segments.append(segment)
        diagnostics.append({
            "case": case, "status": "official_bbox_lift", "n_source": len(points),
            "n_frustum": int(proposal_mask.sum()), "n_points": len(proposal),
            "diagonal": diagonal, "span": span.tolist(), "bbox": [float(v) for v in (cx, cy, width, height)],
            "plane": segment["plane"],
        })
    return segments, diagnostics


def geometric_dummy_segments(files: list[Path]) -> tuple[list[dict], list[dict]]:
    """Extract the 1.4 m elevated connected component from each original scan."""
    o3d.utility.random.seed(31)
    segments: list[dict] = []
    diagnostics: list[dict] = []
    for cache_path in files:
        case = parse_case(cache_path)
        points, intensity = load_original_dummy(case)
        slant = np.linalg.norm(points, axis=1)
        shell_mask = (slant >= 1.80) & (slant <= 3.90)
        shell, shell_intensity = points[shell_mask], intensity[shell_mask]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shell))
        plane, _ = pcd.segment_plane(distance_threshold=0.018, ransac_n=3, num_iterations=2400)
        normal = np.asarray(plane[:3])
        elevation = np.abs(shell @ normal + plane[3])
        elevated = (elevation >= 0.028) & (elevation <= 0.34)
        candidates_points, candidates_intensity = strongest_per_voxel(
            shell[elevated], shell_intensity[elevated], 0.009
        )
        labels = DBSCAN(eps=0.052, min_samples=6, n_jobs=-1).fit_predict(candidates_points)
        candidates: list[dict] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            member = labels == label
            cluster = candidates_points[member]
            if len(cluster) < 70:
                continue
            span = np.ptp(cluster, axis=0)
            diagonal = float(np.linalg.norm(span))
            if not 0.72 <= diagonal <= 1.75:
                continue
            cluster_height = np.abs(cluster @ normal + plane[3])
            size_weight = math.exp(-0.5 * ((diagonal - 1.35) / 0.32) ** 2)
            relief_weight = np.clip(float(np.quantile(cluster_height, 0.9)) / 0.16, 0.35, 1.4)
            score = len(cluster) * (0.30 + 0.70 * size_weight) * relief_weight
            candidates.append({
                "member": member, "points": cluster, "span": span,
                "diagonal": diagonal, "score": float(score),
            })
        if not candidates:
            diagnostics.append({"case": case, "status": "no_geometric_dummy_candidate"})
            continue
        best = max(candidates, key=lambda candidate: candidate["score"])
        member = best["member"]
        segment = {
            "case": case,
            "points": candidates_points[member],
            "intensity": candidates_intensity[member],
            "score": best["score"],
            "diagonal": best["diagonal"],
            "center_range": float(np.linalg.norm(np.median(best["points"], axis=0))),
            "span": best["span"],
            "candidate_count": len(candidates),
            "plane": [float(value) for value in plane],
        }
        segments.append(segment)
        diagnostics.append({
            "case": case, "status": "selected_geometrically", "n_points": len(segment["points"]),
            "diagonal": segment["diagonal"], "score": segment["score"],
            "candidate_count": len(candidates), "plane": segment["plane"],
        })
    return segments, diagnostics


def strongest_per_voxel(points: np.ndarray, intensity: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Retain the strongest return in each voxel without averaging multipath."""
    cells = np.floor(points / voxel).astype(np.int64)
    order = np.argsort(intensity)[::-1]
    _, first = np.unique(cells[order], axis=0, return_index=True)
    chosen = order[first]
    return points[chosen], intensity[chosen]


def prepare_scan(path: Path) -> dict:
    points, intensity = load_scan(path)
    slant_range = np.linalg.norm(points, axis=1)
    # A 1.4 m articulated target spans much more slant range than the chair.
    shell_mask = np.abs(slant_range - 3.0) <= 0.95
    shell = points[shell_mask]
    shell_intensity = intensity[shell_mask]
    if len(shell) < 100:
        raise RuntimeError(f"{path.name}: insufficient points in 3 m shell")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shell))
    plane, _ = pcd.segment_plane(distance_threshold=0.025, ransac_n=3, num_iterations=2000)
    # The released dummy is lying on the seabed.  Do not remove this plane here:
    # doing so also removes the contact-side torso and limbs.  The stationary
    # seabed is removed across time in temporal_dummy_segments instead.
    shell, shell_intensity = strongest_per_voxel(shell, shell_intensity, 0.012)
    return {
        "case": parse_case(path),
        "points": shell,
        "intensity": shell_intensity,
        "plane": [float(value) for value in plane],
    }


def temporal_dummy_segments(files: list[Path]) -> tuple[list[dict], list[dict]]:
    """Remove persistent geometry and choose the human-scale transient per scan."""
    o3d.utility.random.seed(17)
    scans = [prepare_scan(path) for path in files]
    trees = [cKDTree(scan["points"]) for scan in scans]
    segments: list[dict] = []
    diagnostics: list[dict] = []

    for scan_index, scan in enumerate(scans):
        points = scan["points"]
        support = np.zeros(len(points), dtype=np.int16)
        for other_index, tree in enumerate(trees):
            if scan_index == other_index:
                continue
            distance, _ = tree.query(points, k=1, workers=-1)
            support += distance < 0.022
        transient_mask = support <= 2
        transient = points[transient_mask]
        transient_intensity = scan["intensity"][transient_mask]
        labels = DBSCAN(eps=0.050, min_samples=5, n_jobs=-1).fit_predict(transient)

        candidates: list[dict] = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            cluster = transient[mask]
            if len(cluster) < 35:
                continue
            span = np.ptp(cluster, axis=0)
            diagonal = float(np.linalg.norm(span))
            center = np.median(cluster, axis=0)
            center_range = float(np.linalg.norm(center))
            # In the released Cartesian frame the dummy returns in these 3 m
            # acquisitions centre around 2.2--2.4 m slant range; the nominal
            # setting includes the sonar-to-seabed geometry, not body centroid.
            if not (0.55 <= diagonal <= 1.75 and 1.85 <= center_range <= 2.85):
                continue
            size_weight = math.exp(-0.5 * ((diagonal - 1.28) / 0.38) ** 2)
            range_weight = math.exp(-0.5 * ((center_range - 2.28) / 0.32) ** 2)
            compactness = len(cluster) / max(float(np.prod(np.maximum(span, 0.08))), 1e-6)
            compactness_weight = min(1.0, compactness / 1600.0)
            score = len(cluster) * (0.25 + 0.75 * size_weight) * range_weight * (0.6 + 0.4 * compactness_weight)
            candidates.append({
                "mask": mask,
                "n": len(cluster),
                "span": span,
                "diagonal": diagonal,
                "center_range": center_range,
                "score": float(score),
            })
        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
        if not candidates:
            diagnostics.append({
                "case": scan["case"],
                "status": "no_human_scale_transient",
                "n_shell": len(points),
                "n_transient": len(transient),
                "plane": scan["plane"],
            })
            continue
        best = candidates[0]
        segment = {
            "case": scan["case"],
            "points": transient[best["mask"]],
            "intensity": transient_intensity[best["mask"]],
            "score": best["score"],
            "diagonal": best["diagonal"],
            "center_range": best["center_range"],
            "span": best["span"],
            "candidate_count": len(candidates),
            "plane": scan["plane"],
        }
        segments.append(segment)
        diagnostics.append({
            "case": scan["case"],
            "status": "selected",
            "n_shell": len(points),
            "n_transient": len(transient),
            "n_points": best["n"],
            "score": best["score"],
            "diagonal": best["diagonal"],
            "center_range": best["center_range"],
            "span": best["span"].tolist(),
            "candidate_count": len(candidates),
            "plane": scan["plane"],
        })
    return segments, diagnostics


def plane_frame(points: np.ndarray, plane: list[float], reference_normal: np.ndarray) -> np.ndarray:
    """Map sonar XYZ to seabed-plane XY plus physical height above the floor."""
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal @ reference_normal < 0:
        normal = -normal
        offset = -float(plane[3])
    else:
        offset = float(plane[3])
    u = np.array([1.0, 0.0, 0.0]) - reference_normal * reference_normal[0]
    u /= np.linalg.norm(u)
    v = np.cross(reference_normal, u)
    xy = np.column_stack((points @ u, points @ v))
    height = np.abs(points @ normal + offset)
    return np.column_stack((xy, height))


def planar_register(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate rotation and translation in the seabed plane without scaling."""
    source_xy = source[:, :2]
    target_xy = target[:, :2]
    source_center = np.median(source_xy, axis=0)
    target_center = np.median(target_xy, axis=0)
    source_zero = source_xy - source_center
    target_zero = target_xy - target_center
    tree = cKDTree(target_zero)

    def error(angle: float) -> float:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s), (s, c)))
        distances, _ = tree.query(source_zero @ rotation.T, k=1, workers=-1)
        count = max(40, int(len(distances) * 0.72))
        return float(np.mean(np.partition(distances, count - 1)[:count]))

    scored = [(error(math.radians(degrees)), math.radians(degrees)) for degrees in range(0, 360, 5)]
    _, angle = min(scored)
    for step in (1.0, 0.25):
        candidates = angle + np.deg2rad(np.arange(-6, 7) * step)
        _, angle = min((error(candidate), candidate) for candidate in candidates)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s), (s, c)))
    translation = target_center - rotation @ source_center
    aligned = source_xy @ rotation.T + translation

    target_tree = cKDTree(target_xy)
    for _ in range(45):
        distances, indices = target_tree.query(aligned, k=1, workers=-1)
        threshold = min(0.10, float(np.quantile(distances, 0.78)))
        keep = distances <= threshold
        if keep.sum() < 40:
            break
        current = aligned[keep]
        matched = target_xy[indices[keep]]
        current_center = current.mean(axis=0)
        matched_center = matched.mean(axis=0)
        covariance = (current - current_center).T @ (matched - matched_center)
        left, _, right_t = np.linalg.svd(covariance)
        delta_rotation = right_t.T @ left.T
        if np.linalg.det(delta_rotation) < 0:
            right_t[-1] *= -1
            delta_rotation = right_t.T @ left.T
        delta_translation = matched_center - delta_rotation @ current_center
        aligned = aligned @ delta_rotation.T + delta_translation
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation

    distances, _ = target_tree.query(aligned, k=1, workers=-1)
    cutoff = min(0.10, float(np.quantile(distances, 0.78)))
    inliers = distances <= cutoff
    registered = np.column_stack((aligned, source[:, 2]))
    return registered, {
        "yaw_deg": float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])) % 360.0),
        "inlier_fraction": float(np.mean(inliers)),
        "trimmed_rmse_m": float(np.sqrt(np.mean(distances[inliers] ** 2))),
        "rotation_2d": rotation.tolist(),
        "translation_2d": translation.tolist(),
    }


def register_in_seabed_plane(segments: list[dict]) -> tuple[np.ndarray, list[dict], list[str], list[dict]]:
    """Fuse only near-full-length dummy observations using physical SE(2) motion."""
    eligible = [segment for segment in segments if segment["diagonal"] >= 1.02 and len(segment["points"]) >= 1200]
    if len(eligible) < 2:
        raise RuntimeError("Fewer than two full-length dummy observations are available")
    reference = max(eligible, key=lambda segment: segment["score"])
    reference_normal = np.asarray(reference["plane"][:3], dtype=np.float64)
    reference_normal /= np.linalg.norm(reference_normal)
    reference_points = plane_frame(reference["points"], reference["plane"], reference_normal)
    reference_center = np.median(reference_points[:, :2], axis=0)
    reference_points[:, :2] -= reference_center
    reference_sensor = plane_frame(np.zeros((1, 3)), reference["plane"], reference_normal)[0]
    reference_sensor[:2] -= reference_center

    registered = [reference_points]
    observations = [{"case": reference["case"], "points": reference_points, "sensor": reference_sensor}]
    metrics = [{
        "case": reference["case"], "reference": True, "accepted": True,
        "inlier_fraction": 1.0, "trimmed_rmse_m": 0.0, "yaw_deg": 0.0,
    }]
    accepted_cases = [reference["case"]]
    for segment in eligible:
        if segment is reference:
            continue
        source = plane_frame(segment["points"], segment["plane"], reference_normal)
        transformed, fit = planar_register(source, reference_points)
        accepted = fit["inlier_fraction"] >= 0.70 and fit["trimmed_rmse_m"] <= 0.015
        fit.update({"case": segment["case"], "reference": False, "accepted": accepted})
        metrics.append(fit)
        if accepted:
            registered.append(transformed)
            accepted_cases.append(segment["case"])
            source_sensor = plane_frame(np.zeros((1, 3)), segment["plane"], reference_normal)[0]
            rotation = np.asarray(fit["rotation_2d"])
            translation = np.asarray(fit["translation_2d"])
            source_sensor[:2] = rotation @ source_sensor[:2] + translation
            observations.append({"case": segment["case"], "points": transformed, "sensor": source_sensor})
    fused = voxel_down(np.vstack(registered), 0.008)
    return fused, metrics, accepted_cases, observations


def reconstruct_oriented_surface(observations: list[dict]) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, dict]:
    """Fuse view-oriented surfels into a measurement-supported 3-D envelope."""
    oriented_clouds: list[o3d.geometry.PointCloud] = []
    for observation in observations:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observation["points"]))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.060, max_nn=55))
        cloud.orient_normals_towards_camera_location(np.asarray(observation["sensor"]))
        oriented_clouds.append(cloud)
    combined = oriented_clouds[0]
    for cloud in oriented_clouds[1:]:
        combined += cloud
    combined = combined.voxel_down_sample(0.007)
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=22, std_ratio=2.4)
    measured = np.asarray(combined.points)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        combined, depth=9, scale=1.035, linear_fit=True, n_threads=-1
    )
    densities = np.asarray(densities)
    vertices = np.asarray(mesh.vertices)
    distance, _ = cKDTree(measured).query(vertices, k=1, workers=-1)
    density_floor = float(np.quantile(densities, 0.025))
    robust_low = np.quantile(measured, 0.002, axis=0) - np.array((0.025, 0.025, 0.018))
    robust_high = np.quantile(measured, 0.998, axis=0) + np.array((0.025, 0.025, 0.018))
    outside = np.any((vertices < robust_low) | (vertices > robust_high), axis=1)
    mesh.remove_vertices_by_mask((densities < density_floor) | (distance > 0.052) | outside)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters, counts = np.asarray(clusters), np.asarray(counts)
        minimum = max(35, int(counts.max() * 0.012))
        mesh.remove_triangles_by_mask(counts[clusters] < minimum)
        mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=3, lambda_filter=0.38, mu=-0.41)
    mesh.compute_vertex_normals()
    return mesh, measured, {
        "representation": "view-oriented multi-scan screened Poisson envelope",
        "virtual_viewpoints": len(observations),
        "oriented_surfels": int(len(measured)),
        "measurement_support_radius_m": 0.052,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }


def surface_metrics(mesh: o3d.geometry.TriangleMesh, observed: np.ndarray) -> tuple[dict, np.ndarray]:
    sampled = np.asarray(mesh.sample_points_uniformly(number_of_points=50000).points)
    observed_distance, _ = cKDTree(sampled).query(observed, k=1, workers=-1)
    surface_distance, _ = cKDTree(observed).query(sampled, k=1, workers=-1)
    return {
        "observed_to_surface_median_mm": float(np.median(observed_distance) * 1000),
        "observed_to_surface_p95_mm": float(np.quantile(observed_distance, 0.95) * 1000),
        "surface_supported_within_40mm": float(np.mean(surface_distance <= 0.040)),
    }, sampled


def reconstruct_height_surface(fused: np.ndarray, spacing: float = 0.012) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Triangulate the measured top surface; do not synthesize the underside."""
    xy = fused[:, :2]
    height = fused[:, 2]
    low = np.quantile(xy, 0.005, axis=0) - spacing
    high = np.quantile(xy, 0.995, axis=0) + spacing
    xs = np.arange(low[0], high[0] + spacing, spacing)
    ys = np.arange(low[1], high[1] + spacing, spacing)
    nx, ny = len(xs), len(ys)
    sums = np.zeros((ny, nx), dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=np.float64)
    ix = np.clip(np.rint((xy[:, 0] - xs[0]) / spacing).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((xy[:, 1] - ys[0]) / spacing).astype(int), 0, ny - 1)
    np.add.at(sums, (iy, ix), height)
    np.add.at(counts, (iy, ix), 1.0)
    smoothed_weight = gaussian_filter(counts, sigma=1.35)
    smoothed_height = gaussian_filter(sums, sigma=1.35) / np.maximum(smoothed_weight, 1e-9)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distance, nearest = cKDTree(xy).query(grid_xy, k=1, workers=-1)
    supported = (distance.reshape(ny, nx) <= 0.045) & (smoothed_weight >= 0.08)
    supported = binary_closing(supported, iterations=2)
    supported = binary_fill_holes(supported)
    component_map, component_count = connected_labels(supported)
    if component_count:
        component_sizes = np.bincount(component_map.ravel())
        component_sizes[0] = 0
        supported = component_map == int(component_sizes.argmax())
    # Blend the locally averaged height with the nearest measured return so the
    # grid remains faithful at thin limbs and silhouette boundaries.
    nearest_height = height[nearest].reshape(ny, nx)
    z = 0.72 * smoothed_height + 0.28 * nearest_height

    index = -np.ones((ny, nx), dtype=np.int64)
    valid_y, valid_x = np.nonzero(supported)
    vertices = np.column_stack((xs[valid_x], ys[valid_y], z[valid_y, valid_x]))
    index[valid_y, valid_x] = np.arange(len(vertices))
    triangles: list[tuple[int, int, int]] = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            corners = index[row:row + 2, col:col + 2]
            if np.all(corners >= 0):
                a, b = int(corners[0, 0]), int(corners[0, 1])
                c, d = int(corners[1, 0]), int(corners[1, 1])
                triangles.extend(((a, b, d), (a, d, c)))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        clusters, triangle_counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        triangle_counts = np.asarray(triangle_counts)
        minimum = max(20, int(triangle_counts.max() * 0.01))
        mesh.remove_triangles_by_mask(triangle_counts[clusters] < minimum)
        mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5, lambda_filter=0.40, mu=-0.43)
    mesh.compute_vertex_normals()
    return mesh, {
        "representation": "measured seabed-referenced top-surface height field",
        "grid_spacing_m": spacing,
        "underside_completed": False,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }


def ellipsoid(center: tuple[float, float, float], radii: tuple[float, float, float], color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=24)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * np.asarray(radii))
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    return mesh


def tapered_limb(start: np.ndarray, end: np.ndarray, radius_start: float, radius_end: float, color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    vector = end - start
    length = float(np.linalg.norm(vector))
    # Open3D 0.16 has no truncated-cone primitive.  The mean-radius cylinder,
    # blended into ellipsoidal joints, preserves the intended segment envelope.
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=(radius_start + radius_end) / 2.0,
        height=length,
        resolution=20,
        split=2,
    )
    direction = vector / length
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > 1e-9:
        angle = math.acos(float(np.clip(z_axis @ direction, -1.0, 1.0)))
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(cross / cross_norm * angle), center=(0, 0, 0))
    elif direction[2] < 0:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle((math.pi, 0, 0)), center=(0, 0, 0))
    mesh.translate((start + end) / 2.0)
    mesh.paint_uniform_color(color)
    return mesh


def make_dummy_mesh() -> o3d.geometry.TriangleMesh:
    """Create a simplified anatomical dummy with the published 1.4 m height."""
    shell = (0.63, 0.69, 0.75)
    joint = (0.52, 0.59, 0.67)
    mesh = ellipsoid((0.0, 0.0, 0.84), (0.18, 0.115, 0.27), shell)  # torso
    mesh += ellipsoid((0.0, 0.0, 0.605), (0.14, 0.105, 0.13), shell)  # pelvis
    mesh += tapered_limb(np.array([0.0, 0.0, 1.085]), np.array([0.0, 0.0, 1.17]), 0.065, 0.055, joint)
    mesh += ellipsoid((0.0, 0.0, 1.28), (0.105, 0.095, 0.12), shell)  # published top height: 1.40 m

    for side in (-1.0, 1.0):
        shoulder = np.array([side * 0.17, 0.0, 1.015])
        elbow = np.array([side * 0.235, 0.012, 0.78])
        wrist = np.array([side * 0.245, 0.020, 0.56])
        mesh += ellipsoid(tuple(shoulder), (0.075, 0.075, 0.085), joint)
        mesh += tapered_limb(shoulder, elbow, 0.066, 0.052, shell)
        mesh += ellipsoid(tuple(elbow), (0.057, 0.052, 0.060), joint)
        mesh += tapered_limb(elbow, wrist, 0.050, 0.038, shell)
        mesh += ellipsoid(tuple(wrist + np.array([0.0, 0.0, -0.035])), (0.043, 0.034, 0.065), joint)

        hip = np.array([side * 0.083, 0.0, 0.58])
        knee = np.array([side * 0.087, 0.0, 0.315])
        ankle = np.array([side * 0.087, 0.0, 0.065])
        mesh += tapered_limb(hip, knee, 0.076, 0.058, shell)
        mesh += ellipsoid(tuple(knee), (0.061, 0.057, 0.065), joint)
        mesh += tapered_limb(knee, ankle, 0.056, 0.040, shell)
        mesh += ellipsoid((side * 0.087, -0.035, 0.035), (0.052, 0.095, 0.035), joint)

    mesh.compute_vertex_normals()
    return mesh


def fit_segment_to_prior(points: np.ndarray, model_points: np.ndarray) -> tuple[np.ndarray, dict]:
    source = voxel_down(points, 0.012)
    target = voxel_down(model_points, 0.008)
    source_center, source_axes = principal_axes(source)
    target_center, target_axes = principal_axes(target)
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.075, max_nn=50))

    best = None
    best_quality = (-1.0, float("inf"))
    for signed_permutation in signed_permutation_matrices():
        rotation = target_axes @ signed_permutation @ source_axes.T
        initial = np.eye(4)
        initial[:3, :3] = rotation
        initial[:3, 3] = target_center - rotation @ source_center
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.14,
            initial,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=0.070)
            ),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=90),
        )
        quality = (float(result.fitness), float(result.inlier_rmse))
        if quality[0] > best_quality[0] + 1e-6 or (
            abs(quality[0] - best_quality[0]) <= 1e-6 and quality[1] < best_quality[1]
        ):
            best, best_quality = result, quality
    assert best is not None
    transformed = points @ best.transformation[:3, :3].T + best.transformation[:3, 3]
    return transformed, {
        "fitness": float(best.fitness),
        "rmse": float(best.inlier_rmse),
        "transform_to_canonical": best.transformation.tolist(),
    }


def reconstruct_measured_surface(fused: np.ndarray) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Build the delivered surface from registered sonar points, not the prior.

    Screened Poisson supplies local continuity where the sonar sampling pattern
    has small gaps.  Density, measurement-distance, and component tests prevent
    it from inventing unsupported sheets away from the measured returns.
    """
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused))
    cloud, kept = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.6)
    clean = np.asarray(cloud.points)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.075, max_nn=60))
    cloud.orient_normals_consistent_tangent_plane(28)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=9, scale=1.04, linear_fit=True, n_threads=-1
    )
    densities = np.asarray(densities)

    # Remove the weakest Poisson extrapolation and any surface farther than two
    # to four sonar resolution cells from actual registered measurements.
    vertices = np.asarray(mesh.vertices)
    distance, _ = cKDTree(clean).query(vertices, k=1, workers=-1)
    density_floor = float(np.quantile(densities, 0.035))
    unsupported = (densities < density_floor) | (distance > 0.070)
    mesh.remove_vertices_by_mask(unsupported)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        counts = np.asarray(counts)
        minimum = max(40, int(counts.max() * 0.012))
        mesh.remove_triangles_by_mask(counts[clusters] < minimum)
        mesh.remove_unreferenced_vertices()
    mesh = mesh.filter_smooth_taubin(number_of_iterations=4, lambda_filter=0.45, mu=-0.48)
    mesh.compute_vertex_normals()
    return mesh, {
        "input_fused_points": int(len(fused)),
        "outlier_filtered_points": int(len(clean)),
        "poisson_density_floor": density_floor,
        "measurement_support_radius_m": 0.070,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }


def add_shaded_mesh(ax, mesh: o3d.geometry.TriangleMesh, color: tuple[float, float, float]) -> np.ndarray:
    """Draw a continuous shaded mesh while keeping report rendering lightweight."""
    shown = mesh
    if len(mesh.triangles) > 10000:
        shown = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
    vertices = np.asarray(shown.vertices)
    triangles = np.asarray(shown.triangles)
    faces = vertices[triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    illumination = np.clip(normals @ light, 0.0, 1.0)
    shade = 0.42 + 0.58 * illumination
    face_colors = np.clip(np.asarray(color)[None, :] * shade[:, None], 0.0, 1.0)
    collection = Poly3DCollection(faces, facecolors=face_colors, edgecolors="none", alpha=0.96)
    ax.add_collection3d(collection)
    return vertices


def render_segmentation_grid(segments: list[dict], output: Path) -> None:
    fig = plt.figure(figsize=(16, 8.5), facecolor="#08111f")
    fig.suptitle(
        "SUOP Dummy — temporal-background-subtracted proposals (3 m scans)",
        color="white", fontsize=20, fontweight="bold", y=0.96,
    )
    for index, segment in enumerate(segments, start=1):
        ax = fig.add_subplot(2, 4, index, projection="3d")
        points = segment["points"]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="turbo", s=1.2, linewidths=0)
        configure_3d_axis(ax, points, 18, -58)
        ax.set_title(f"{segment['case']}  •  {len(points):,} pts", color="#dbeafe", fontsize=11, fontweight="bold")
    fig.text(
        0.5, 0.025,
        "Persistent seabed/test-rig geometry is removed; each panel is the best 1.4 m dummy-scale moving cluster.",
        color="#8fa6bf", ha="center", fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.90, wspace=0.02, hspace=0.13)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_reconstruction(mesh: o3d.geometry.TriangleMesh, fused: np.ndarray, best_raw: np.ndarray, accepted_cases: list[str], metrics: dict, output: Path) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor="#08111f")
    fig.suptitle(
        "SUOP 1.4 m Dummy — view-aware sonar surface reconstruction",
        color="white", fontsize=22, fontweight="bold", y=0.96,
    )
    mesh_vertices = np.asarray(mesh.vertices)
    panels = (
        ("points", best_raw, None, "Best seabed-separated sonar observation", "#38bdf8"),
        ("points", fused, None, f"Rigidly registered sonar evidence ({len(accepted_cases)} poses)", "#22d3ee"),
        ("mesh", mesh_vertices, None, "Continuous surface reconstructed from sonar", "#d7e0ea"),
        ("mesh", mesh_vertices, fused, "Evidence overlay on data-derived surface", "#94a3b8"),
    )
    for index, (kind, base, overlay, title, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        if kind == "mesh":
            shown = add_shaded_mesh(ax, mesh, (0.72, 0.78, 0.85))
        else:
            shown = base
            if len(shown) > 16000:
                shown = shown[np.random.default_rng(17).choice(len(shown), 16000, replace=False)]
            ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], s=1.2, c=color, alpha=0.82, linewidths=0)
        extent = shown
        if overlay is not None and len(overlay):
            overlay_shown = overlay[::2]
            ax.scatter(overlay_shown[:, 0], overlay_shown[:, 1], overlay_shown[:, 2], s=4.0, c="#f97316", alpha=0.80, linewidths=0)
            extent = np.vstack((shown, overlay))
        configure_3d_axis(ax, extent, 72, -72)
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


def render_turntable(mesh: o3d.geometry.TriangleMesh, fused: np.ndarray, output: Path) -> None:
    fig = plt.figure(figsize=(16, 5.5), facecolor="#08111f")
    vertices = np.asarray(mesh.vertices)
    for index, azimuth in enumerate((-45, 45, 135, 225), start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        add_shaded_mesh(ax, mesh, (0.72, 0.78, 0.85))
        ax.scatter(fused[::2, 0], fused[::2, 1], fused[::2, 2], s=3.0, c="#f97316", alpha=0.74, linewidths=0)
        configure_3d_axis(ax, vertices, 32, azimuth)
        ax.set_title(f"Azimuth {azimuth % 360}°", color="#dbeafe", fontsize=11, fontweight="bold")
    fig.suptitle("Data-derived dummy surface (grey) with registered sonar evidence (orange)", color="white", fontsize=17, fontweight="bold", y=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("segment", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(CLEAN_DIR.glob("dummy__dummy_range_3m__case_*_clean.npz"))
    if not files:
        raise FileNotFoundError("No cached 3 m dummy scans found")
    segments, manifest = geometric_dummy_segments(files)
    if not segments:
        raise RuntimeError("Temporal subtraction found no human-scale moving clusters")
    for segment in segments:
        np.savez_compressed(
            args.output_dir / f"{segment['case']}_dummy_temporal_segment.npz",
            xyz=segment["points"].astype(np.float32),
            intensity=segment["intensity"].astype(np.float32),
        )
        print(
            f"{segment['case']}: dummy={len(segment['points']):,} diag={segment['diagonal']:.3f}m "
            f"range={segment['center_range']:.3f}m score={segment['score']:.1f}", flush=True,
        )
    render_segmentation_grid(segments, args.output_dir / "dummy_temporal_segments.png")
    (args.output_dir / "segmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.stage == "segment":
        print(args.output_dir / "dummy_temporal_segments.png")
        return

    fused, registration_metrics, accepted_cases, observations = register_in_seabed_plane(segments)
    for fit in registration_metrics:
        print(
            f"planar fit {fit['case']}: accepted={fit['accepted']} "
            f"inliers={fit['inlier_fraction']:.3f} rmse={fit['trimmed_rmse_m']*1000:.1f}mm "
            f"yaw={fit['yaw_deg']:.1f}deg",
            flush=True,
        )
    height_mesh, height_stats = reconstruct_height_surface(fused)
    oriented_mesh, oriented_surfels, oriented_stats = reconstruct_oriented_surface(observations)
    if not len(oriented_mesh.triangles):
        raise RuntimeError("View-oriented reconstruction produced no supported triangles")
    oriented_quality, _ = surface_metrics(oriented_mesh, fused)
    height_quality, _ = surface_metrics(height_mesh, fused)
    reconstructed_mesh = oriented_mesh
    surface_stats = oriented_stats
    selected_quality = oriented_quality
    metrics = {
        "method": "geometric seabed separation + constrained SE(2) registration + view-oriented screened Poisson fusion",
        "delivered_geometry_source": "registered sonar measurements",
        "anatomical_template_used": False,
        "physical_observability": "measured multi-view envelope; fully occluded regions are not guaranteed",
        "published_dummy_length_m": 1.4,
        "accepted_cases": accepted_cases,
        "observed_points_fused": len(fused),
        **selected_quality,
        "surface_reconstruction": surface_stats,
        "candidate_comparison": {
            "view_oriented_poisson": oriented_quality,
            "height_field_baseline": height_quality,
        },
        "planar_fits": registration_metrics,
    }
    observed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused))
    observed.paint_uniform_color((0.10, 0.75, 0.88))
    o3d.io.write_point_cloud(str(args.output_dir / "dummy_observed_fused.ply"), observed)
    height_mesh.paint_uniform_color((0.52, 0.60, 0.68))
    o3d.io.write_triangle_mesh(str(args.output_dir / "dummy_heightfield_baseline.ply"), height_mesh, write_vertex_colors=True)
    reconstructed_mesh.paint_uniform_color((0.72, 0.78, 0.85))
    o3d.io.write_triangle_mesh(str(args.output_dir / "dummy_reconstruction.ply"), reconstructed_mesh, write_vertex_colors=True)
    (args.output_dir / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    reference_case = next(fit["case"] for fit in registration_metrics if fit["reference"])
    best_segment = next(segment for segment in segments if segment["case"] == reference_case)
    reference_normal = np.asarray(best_segment["plane"][:3], dtype=np.float64)
    reference_normal /= np.linalg.norm(reference_normal)
    best_raw = plane_frame(best_segment["points"], best_segment["plane"], reference_normal)
    best_raw[:, :2] -= np.median(best_raw[:, :2], axis=0)
    render_reconstruction(reconstructed_mesh, fused, best_raw, accepted_cases, metrics, args.output_dir / "dummy_reconstruction.png")
    render_turntable(reconstructed_mesh, fused, args.output_dir / "dummy_reconstruction_turntable.png")
    print(args.output_dir / "dummy_reconstruction.png")


if __name__ == "__main__":
    main()
