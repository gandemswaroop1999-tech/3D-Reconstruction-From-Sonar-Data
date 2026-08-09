import json
from pathlib import Path

ROUTE_A = Path(__file__).resolve().parents[1] / "Route_A_Geometric_Fusion"
paths = [
    (str(ROUTE_A / "05_variant_viewaware_seabed_plane/chair_viewaware_outputs/reconstruction_metrics.json"), "Chair Route-A(viewaware)"),
    (str(ROUTE_A / "05_variant_viewaware_seabed_plane/tyre_viewaware_outputs/reconstruction_metrics.json"), "Tyre Route-A(viewaware)"),
    (str(ROUTE_A / "06_variant_first_return_ray_TSDF/chair_suop_tsdf_outputs/reconstruction_metrics.json"), "Chair Route-B(ray TSDF)"),
    (str(ROUTE_A / "06_variant_first_return_ray_TSDF/tyre_suop_tsdf_outputs/reconstruction_metrics.json"), "Tyre Route-B(ray TSDF)"),
]
for p, label in paths:
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(label, "ERROR", e)
        continue
    print("=" * 70)
    print(label)
    for k in ("method", "object", "accepted_cases", "observed_points_fused",
              "observed_to_surface_median_mm", "observed_to_surface_p95_mm",
              "surface_supported_within_40mm"):
        if k in d:
            print(f"  {k}: {d[k]}")
    sr = d.get("surface_reconstruction", {})
    print("  surface_reconstruction:", json.dumps(sr))
    pf = d.get("planar_fits") or d.get("pose_fits") or []
    n_acc = sum(1 for x in pf if x.get("accepted"))
    print(f"  pose fits: {len(pf)} total, {n_acc} accepted")
    # print any other top-level scalar keys
    for k, v in d.items():
        if k in ("method", "object", "accepted_cases", "observed_points_fused",
                 "observed_to_surface_median_mm", "observed_to_surface_p95_mm",
                 "surface_supported_within_40mm", "surface_reconstruction",
                 "planar_fits", "pose_fits"):
            continue
        if isinstance(v, (str, int, float, bool)):
            print(f"  {k}: {v}")
