"""Recognizable hybrid completion for sparse SUOP chair and tyre sonar data.

The full surfaces are dimensionally constrained models; registered sonar points
remain separate evidence.  This is intentionally labelled model-assisted and
does not claim that occluded geometry was directly measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from reconstruct_chair import configure_3d_axis, make_folding_chair_mesh
from reconstruct_dummy import add_shaded_mesh


ROOT = Path(__file__).resolve().parent


def make_tyre_mesh(outer_radius: float, inner_radius: float, width: float) -> o3d.geometry.TriangleMesh:
    """Create a flattened automotive-tyre surface with subtle tread relief."""
    around, cross = 160, 56
    theta = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, cross, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    major = (outer_radius + inner_radius) / 2.0
    radial_half = (outer_radius - inner_radius) / 2.0
    outer_weight = np.maximum(np.cos(pp), 0.0) ** 8
    tread = 0.0055 * outer_weight * (
        0.55 * np.cos(36.0 * tt + 2.0 * np.sin(pp))
        + 0.45 * np.cos(18.0 * tt - 3.0 * pp)
    )
    radius = major + radial_half * np.cos(pp) + tread
    z = (width / 2.0) * np.sin(pp)
    vertices = np.column_stack((
        (radius * np.cos(tt)).ravel(),
        (radius * np.sin(tt)).ravel(),
        z.ravel(),
    ))
    triangles: list[tuple[int, int, int]] = []
    for i in range(around):
        ni = (i + 1) % around
        for j in range(cross):
            nj = (j + 1) % cross
            a, b = i * cross + j, ni * cross + j
            c, d = i * cross + nj, ni * cross + nj
            triangles.extend(((a, b, d), (a, d, c)))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def model_metrics(mesh: o3d.geometry.TriangleMesh, evidence: np.ndarray, support_radius: float) -> dict:
    surface = np.asarray(mesh.sample_points_uniformly(number_of_points=60000).points)
    observed_distance, _ = cKDTree(surface).query(evidence, k=1, workers=-1)
    surface_distance, _ = cKDTree(evidence).query(surface, k=1, workers=-1)
    bounds = np.ptp(np.asarray(mesh.vertices), axis=0)
    return {
        "observed_to_completed_surface_median_mm": float(np.median(observed_distance) * 1000),
        "observed_to_completed_surface_p95_mm": float(np.quantile(observed_distance, 0.95) * 1000),
        "completed_surface_supported_fraction": float(np.mean(surface_distance <= support_radius)),
        "mesh_bounds_m": bounds.tolist(),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }


def render_report(name: str, mesh: o3d.geometry.TriangleMesh, evidence: np.ndarray, metrics: dict, output: Path, elevation: float) -> None:
    fig = plt.figure(figsize=(18, 10), facecolor="#08111f")
    fig.suptitle(
        f"SUOP {name} — recognizable measurement-constrained completion",
        color="white", fontsize=22, fontweight="bold", y=0.96,
    )
    vertices = np.asarray(mesh.vertices)
    panels = (
        ("points", "Registered sonar evidence (measured)"),
        ("mesh", "Dimensionally constrained completed surface (inferred)"),
        ("overlay", "Measured evidence over completed surface"),
        ("overlay", "Audit view: orange is measured; grey is inferred"),
    )
    views = ((elevation, -55), (elevation, -55), (elevation, -55), (elevation, 45))
    for index, ((kind, title), (elev, azim)) in enumerate(zip(panels, views), start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        if kind == "points":
            ax.scatter(evidence[:, 0], evidence[:, 1], evidence[:, 2], s=3.0, c="#22d3ee", alpha=0.88, linewidths=0)
            extent = evidence
        else:
            shown = add_shaded_mesh(ax, mesh, (0.72, 0.78, 0.85))
            extent = shown
            if kind == "overlay":
                ax.scatter(evidence[:, 0], evidence[:, 1], evidence[:, 2], s=5.0, c="#f97316", alpha=0.86, linewidths=0)
                extent = np.vstack((shown, evidence))
        configure_3d_axis(ax, extent, elev, azim)
        ax.set_title(title, color="#e5edf7", fontsize=14, fontweight="bold", pad=12)
    fig.text(
        0.5, 0.025,
        f"Measured→completed median {metrics['observed_to_completed_surface_median_mm']:.1f} mm • "
        f"p95 {metrics['observed_to_completed_surface_p95_mm']:.1f} mm • full surface includes explicitly inferred regions",
        color="#9fb2c8", ha="center", fontsize=11,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.90, wspace=0.02, hspace=0.12)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_turntable(name: str, mesh: o3d.geometry.TriangleMesh, evidence: np.ndarray, output: Path, elevation: float) -> None:
    fig = plt.figure(figsize=(16, 5.2), facecolor="#08111f")
    vertices = np.asarray(mesh.vertices)
    for index, azimuth in enumerate((-45, 45, 135, 225), start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        add_shaded_mesh(ax, mesh, (0.72, 0.78, 0.85))
        ax.scatter(evidence[:, 0], evidence[:, 1], evidence[:, 2], s=4.0, c="#f97316", alpha=0.84, linewidths=0)
        configure_3d_axis(ax, vertices, elevation, azimuth)
        ax.set_title(f"Azimuth {azimuth % 360}°", color="#dbeafe", fontsize=11, fontweight="bold")
    fig.suptitle(f"{name}: inferred completion (grey) with measured sonar (orange)", color="white", fontsize=17, fontweight="bold", y=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def reconstruct_chair() -> None:
    output = ROOT / "chair_recognizable_outputs"
    output.mkdir(parents=True, exist_ok=True)
    evidence_cloud = o3d.io.read_point_cloud(str(ROOT / "chair_outputs" / "chair_observed_fused.ply"))
    evidence = np.asarray(evidence_cloud.points)
    if not len(evidence):
        raise RuntimeError("Canonical chair sonar evidence is missing")
    mesh = make_folding_chair_mesh()
    metrics = model_metrics(mesh, evidence, support_radius=0.020)
    metrics.update({
        "method": "robust sonar registration + dimensionally constrained folding-chair completion",
        "geometry_source": "hybrid: measured sonar evidence plus inferred CAD-like completion",
        "published_dimensions_m": {"length": 0.20, "height": 0.40},
        "free_scale_fitting_used": False,
    })
    mesh.paint_uniform_color((0.72, 0.78, 0.85))
    o3d.io.write_triangle_mesh(str(output / "chair_recognizable_reconstruction.ply"), mesh, write_vertex_colors=True)
    o3d.io.write_point_cloud(str(output / "chair_measured_evidence.ply"), evidence_cloud)
    (output / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_report("Folding Chair", mesh, evidence, metrics, output / "chair_recognizable_reconstruction.png", 18)
    render_turntable("Folding Chair", mesh, evidence, output / "chair_recognizable_turntable.png", 18)


def reconstruct_tyre() -> None:
    output = ROOT / "tyre_recognizable_outputs"
    output.mkdir(parents=True, exist_ok=True)
    viewaware_dir = ROOT.parent / "05_variant_viewaware_seabed_plane" / "tyre_viewaware_outputs"
    evidence_cloud = o3d.io.read_point_cloud(str(viewaware_dir / "tyre_observed_fused.ply"))
    evidence = np.asarray(evidence_cloud.points)
    if not len(evidence):
        raise RuntimeError("Canonical tyre sonar evidence is missing")
    center_xy = np.median(evidence[:, :2], axis=0)
    evidence = evidence.copy()
    evidence[:, :2] -= center_xy
    z_low, z_high = np.quantile(evidence[:, 2], (0.02, 0.98))
    measured_width = float(np.clip(z_high - z_low, 0.16, 0.26))
    center_z = float((z_low + z_high) / 2.0)
    radial = np.linalg.norm(evidence[:, :2], axis=1)
    inner_radius = float(np.clip(np.quantile(radial, 0.08), 0.14, 0.19))
    mesh = make_tyre_mesh(outer_radius=0.33, inner_radius=inner_radius, width=measured_width)
    mesh.translate((0.0, 0.0, center_z))
    # Robustly fit the measured ring to the dimensionally fixed surface.  The
    # fused cloud still contains central seabed and multipath returns, so use a
    # loose first pass followed by a 75 mm model-consistency inlier gate.
    target = mesh.sample_points_uniformly(number_of_points=70000)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.040, max_nn=50))
    preliminary_tree = cKDTree(np.asarray(target.points))
    preliminary_distance, _ = preliminary_tree.query(evidence, k=1, workers=-1)
    preliminary = evidence[preliminary_distance <= 0.12]
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(preliminary))
    result = o3d.pipelines.registration.registration_icp(
        source, target, 0.12, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=0.060)
        ),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=90),
    )
    evidence = evidence @ result.transformation[:3, :3].T + result.transformation[:3, 3]
    final_distance, _ = preliminary_tree.query(evidence, k=1, workers=-1)
    evidence_inliers = evidence[final_distance <= 0.075]
    metrics = model_metrics(mesh, evidence_inliers, support_radius=0.035)
    metrics.update({
        "method": "view-aware sonar registration + measured-profile parametric tyre completion",
        "geometry_source": "hybrid: measured sonar evidence plus inferred parametric completion",
        "published_outer_radius_m": 0.33,
        "inferred_inner_radius_m": inner_radius,
        "measured_width_m": measured_width,
        "free_scale_fitting_used": False,
        "rigid_fit_fitness": float(result.fitness),
        "rigid_fit_rmse_m": float(result.inlier_rmse),
        "input_evidence_points": int(len(evidence)),
        "model_consistent_evidence_points": int(len(evidence_inliers)),
        "outlier_gate_m": 0.075,
    })
    mesh.paint_uniform_color((0.30, 0.34, 0.39))
    canonical_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(evidence_inliers))
    canonical_cloud.paint_uniform_color((0.10, 0.75, 0.88))
    o3d.io.write_triangle_mesh(str(output / "tyre_recognizable_reconstruction.ply"), mesh, write_vertex_colors=True)
    o3d.io.write_point_cloud(str(output / "tyre_measured_evidence.ply"), canonical_cloud)
    (output / "reconstruction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_report("Tyre", mesh, evidence_inliers, metrics, output / "tyre_recognizable_reconstruction.png", 25)
    render_turntable("Tyre", mesh, evidence_inliers, output / "tyre_recognizable_turntable.png", 25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("object", choices=("chair", "tyre", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if args.object in ("chair", "all"):
        reconstruct_chair()
    if args.object in ("tyre", "all"):
        reconstruct_tyre()


if __name__ == "__main__":
    main()
