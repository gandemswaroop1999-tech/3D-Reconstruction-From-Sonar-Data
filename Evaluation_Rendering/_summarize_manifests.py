import json

reg = json.load(open("registration_manifest_v2.json", encoding="utf-8"))
print("=== registration_manifest_v2 ===")
for r in reg:
    print(r["object"], {k: r.get(k) for k in (
        "status", "n_segments", "n_reachable", "n_edges_above_threshold",
        "n_mst_edges_used", "n_loop_closure_edges", "n_edges_total_tested",
        "n_points_fused_raw", "n_points_fused_downsampled", "pairwise_time_s")})

print()
print("=== meshing_manifest ===")
try:
    mesh = json.load(open("meshing_manifest.json", encoding="utf-8"))
    if isinstance(mesh, dict):
        mesh = list(mesh.values())
    for m in mesh:
        print(json.dumps(m, indent=None)[:1200])
except Exception as e:
    print("meshing manifest error:", e)

print()
print("=== partial_fusion_manifest ===")
try:
    pf = json.load(open("partial_fusion_manifest.json", encoding="utf-8"))
    print(json.dumps(pf, indent=1)[:3000])
except Exception as e:
    print("pf error:", e)

print()
print("=== partial_fusion_selection ===")
try:
    sel = json.load(open("partial_fusion_selection.json", encoding="utf-8"))
    print(json.dumps(sel, indent=1)[:2000])
except Exception as e:
    print("sel error:", e)
