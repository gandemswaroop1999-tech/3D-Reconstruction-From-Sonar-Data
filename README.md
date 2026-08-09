# SUOP Sonar 3D Reconstruction — Route A / Route B Run Guide

This workspace reconstructs small underwater objects (**chair, dummy, drum,
tyre/tire, net**) from mechanically-scanned sonar point clouds, along two routes:

* **Route A — Geometric multi-view fusion**: pairwise registration
  (FPFH → RANSAC → ICP) → pose-graph optimisation → cloud fusion → meshing
  (Poisson / ball-pivoting), plus three reconstruction variants.
* **Route B — Learned completion**: synthetic paired-dataset generation →
  Point Completion Network (PCN) training → inference/completion rendering.

Every stage below lists **how to run it**, the **expected output**, and a
**realistic time estimate** (measured on the project machine — RTX 4060 Laptop
GPU, 8 CPU cores — from the saved logs).

---

## 0. Prerequisites

```powershell
# Python environment (conda env at R:\sonar\envs\suop-recon)
R:\sonar\Environment_Setup\env_setup.bat          # create env + install deps
R:\sonar\Environment_Setup\torch_install.bat      # install PyTorch (CUDA)

# Dataset (git-lfs repo with 52k+ LFS files)
R:\sonar\Environment_Setup\checkout.bat           # clone suop_repo
R:\sonar\Environment_Setup\lfs_pull.bat           # pull all LFS blobs
```

The interpreter used everywhere is:

```powershell
R:\sonar\envs\suop-recon\python.exe
```

Every script resolves the project root from `__file__`, so it can be launched
from any working directory. The `.bat` runners already `cd /d "%~dp0"` and log
to `*.log` next to themselves.

---

## 1. Shared preprocessing (both routes)

Run in order — every stage consumes the previous stage's output.

### 1.1 Extraction — `Shared_Preprocessing\01_extraction`

```powershell
R:\sonar\Shared_Preprocessing\01_extraction\build_cache.bat
# or directly:
R:\sonar\envs\suop-recon\python.exe R:\sonar\Shared_Preprocessing\01_extraction\build_cache.py
```

Parses every case's `point_cloud.xyz` + metadata (`case_settings.txt`,
`head_info.txt`, `ping_info.csv`) into compact `.npz` caches (xyz + intensity).

**Output:** `data\cache\*.npz` — 120 case files + `cache_manifest.json`.
**Time:** ~45 s (120 cases, measured 44 s).

### 1.2 Denoising — `Shared_Preprocessing\02_denoising`

```powershell
R:\sonar\Shared_Preprocessing\02_denoising\denoise.bat
```

Statistical Outlier Removal (16 neighbours, 2σ) + 1 cm voxel downsample, keeping
intensity per voxel.

**Output:** `data\cache_clean\*_clean.npz` — 120 files + manifest.
**Time:** ~35 s (measured 34 s).

### 1.3 Segmentation — `Shared_Preprocessing\03_segmentation`

```powershell
R:\sonar\Shared_Preprocessing\03_segmentation\segment.bat      # v1 (DBSCAN only)
R:\sonar\Shared_Preprocessing\03_segmentation\segment_v3.bat   # v3 (RANSAC plane removal + DBSCAN) — used by Route A
```

Removes the seabed plane (v3) and clusters object-scale points with DBSCAN,
keeping clusters near the nominal object sizes (chair 0.4 m, dummy 1.4 m,
drum 1.2 m, tyre 0.66 m, net 0.5 m).

**Output:** `data\cache_segmented\*_seg.npz` (v1) or
`data\cache_segmented_v3\*_seg.npz` (v3) — 120 files each.
**Time:** ~40 s (measured 38 s for v3).

---

## 2. Route A — Geometric multi-view fusion

### 2.1 Pairwise registration — `Route_A_Geometric_Fusion\01_pairwise_registration`

Three experimental variants (`reg_test.py`, `reg_test2.py`, `reg_test3.py`) that
try FPFH features + RANSAC global registration + ICP refinement on every scan
pair. Run each with its `.bat`:

```powershell
R:\sonar\Route_A_Geometric_Fusion\01_pairwise_registration\reg_test2.bat
R:\sonar\Route_A_Geometric_Fusion\01_pairwise_registration\reg_test3.bat
```

**Output:** console/`reg_test*.log` per-pair `ransac_fit`, `icp_fit`,
`icp_rmse`; JSON results in `reg_test_results*.json`.
**Time:** a few minutes per script (8 scans, all pairs).

### 2.2 Pose-graph optimisation — `Route_A_Geometric_Fusion\02_pose_graph_optimization`

```powershell
R:\sonar\Route_A_Geometric_Fusion\02_pose_graph_optimization\multiway_registration.bat     # v1
R:\sonar\Route_A_Geometric_Fusion\02_pose_graph_optimization\multiway_registration_v2.bat  # v2 (+ loop closures, global optimisation)
```

For each of the 5 objects: pairwise-register all segments, keep edges above a
fitness threshold, build an MST pose graph (+ loop-closure edges in v2),
globally optimise, and export the fused cloud. V2 is the production version.

**Output:**
* `data\cache_fused\{object}_fused.npz` (v1) — 5 files
* `data\cache_fused_v2\{object}_fused.npz` (v2) — 5 files
* `registration_manifest.json` / `registration_manifest_v2.json` (segment counts,
  edge stats, per-pair fitness/RMSE)

**Time:** ~45–60 min total for all 5 objects. Pairwise registration dominates —
the largest object (net, 24 segments) took ~16 min of pairwise ICP alone
(measured 922–954 s). Individual small objects take ~5–10 min.

### 2.3 Quality-filtered fusion — `Route_A_Geometric_Fusion\03_fusion`

```powershell
R:\sonar\Route_A_Geometric_Fusion\03_fusion\partial_fusion.bat
```

Recomputes only the edges in the trusted selection
(`partial_fusion_selection.json`) and fuses the kept subset per object.

**Output:** `data\cache_fused_partial\{object}_fused.npz` (5 files) +
`partial_fusion_manifest.json`.
**Time:** ~10–20 min total; per object ~2 min of pairwise recomputation
(measured: chair 125 s, net 96 s) + graph optimisation.

### 2.4 Meshing — `Route_A_Geometric_Fusion\04_meshing`

```powershell
R:\sonar\Route_A_Geometric_Fusion\04_meshing\meshing_pipeline.bat
```

For each fused cloud: normal estimation → **Screened Poisson** (depth 8) and
**Ball-Pivoting** reconstruction, both cleaned (degenerate/duplicate removal,
largest-component keep).

**Output:** `meshes\{object}_{poisson,bpa}.ply` (10 meshes) +
`meshing_manifest.json` (vertex/triangle counts, bounding boxes).
**Time:** ~5–10 min total (Poisson depth 8 is the expensive step).

### 2.5 Variant — view-aware seabed-plane — `Route_A_Geometric_Fusion\05_variant_viewaware_seabed_plane`

```powershell
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\05_variant_viewaware_seabed_plane\reconstruct_viewaware.py chair
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\05_variant_viewaware_seabed_plane\reconstruct_viewaware.py tire
```

Registers the 3 m segmented scans on an SE(2) seabed-plane prior, fuses
registered surfels, meshes, and renders. `chair | tire` are the two supported
objects (no `.bat` — run the script directly).

**Output** (in the same folder): `chair_viewaware_outputs\` / `tyre_viewaware_outputs\`
— `{obj}_segments.png`, `{obj}_observed_fused.ply`, `{obj}_reconstruction.ply`,
`{obj}_reconstruction.png`, `{obj}_reconstruction_turntable.png`,
`reconstruction_metrics.json`, `segmentation_manifest.json`.
**Time:** ~10–20 min per object (registration + surfel fusion + meshing).

### 2.6 Variant — first-return ray TSDF — `Route_A_Geometric_Fusion\06_variant_first_return_ray_TSDF`

```powershell
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\06_variant_first_return_ray_TSDF\reconstruct_suop_tsdf.py both
# object arg: chair | tire | both
```

No model prior: treats every registered return as the end of an acoustic ray,
fuses the resulting signed distances in a dense TSDF, extracts the surface via
marching cubes, and colour-codes mesh vertices by measured-return support.

**Output** (in the same folder): `chair_suop_tsdf_outputs\` / `tyre_suop_tsdf_outputs\`
— `{obj}_registered_returns.ply`, `{obj}_ray_tsdf.ply`, `{obj}_ray_tsdf.png`,
`{obj}_ray_tsdf_turntable.png`, `reconstruction_metrics.json`,
`segmentation_manifest.json`.
**Time:** ~10–20 min per object.

### 2.7 Variant — model-assisted hybrid — `Route_A_Geometric_Fusion\07_variant_model_assisted_hybrid`

```powershell
# Chair: temporal segmentation + template-constrained completion
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\07_variant_model_assisted_hybrid\reconstruct_chair.py --stage all

# Dummy: same, with a dimensionally-correct mannequin prior + height-field baseline
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\07_variant_model_assisted_hybrid\reconstruct_dummy.py --stage all

# "Recognizable" report versions (chair + tyre) — require stage-05/07 outputs first
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_A_Geometric_Fusion\07_variant_model_assisted_hybrid\reconstruct_recognizable.py all
```

`--stage segment` runs only the temporal segmentation; `--stage all` runs
segmentation + reconstruction. These read the official SUOP bbox labels
(`suop_detection_reference\...\bbox_labels\`) for the benchmark view, the
original clouds (`suop_repo\SUOP_dataset\...`) and `data\cache_clean`.

**Output** (in the same folder):
* `chair_outputs\` — `chair_temporal_segments.png`, `case_*_chair_temporal_segment.npz`,
  `chair_observed_fused.ply`, `chair_reconstruction.ply`, `chair_reconstruction.png`,
  `chair_reconstruction_turntable.png`, `reconstruction_metrics.json`, `segmentation_manifest.json`
* `dummy_outputs\` — same pattern + `dummy_heightfield_baseline.ply`
* `chair_recognizable_outputs\` / `tyre_recognizable_outputs\` —
  `{obj}_measured_evidence.ply`, `{obj}_recognizable_reconstruction.ply`,
  `{obj}_recognizable_reconstruction.png`, `{obj}_recognizable_turntable.png`,
  `reconstruction_metrics.json`

**Time:** ~5–15 min per object (chair/dummy); recognizable variants ~2–5 min each.

---

## 3. Route B — Learned PCN completion

### 3.1 Simulation dataset generation — `Route_B_Learned_Completion\01_simulation_dataset_generation`

```powershell
R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_chair.bat        # synthetic chair → dataset_chair.npz
R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_dummy.bat        # synthetic dummy → dataset_dummy.npz
R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_dummy_suop.bat   # from Route A dummy mesh → dataset_dummy_suop.npz
R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_drum_suop.bat    # from Route A drum mesh → dataset_drum_suop.npz
# direct (all args optional):
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_dataset.py --n 6000 --procs 8 --out dataset_chair.npz
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\gen_suop_dummy_dataset.py --mesh R:\sonar\Route_A_Geometric_Fusion\04_meshing\meshes\dummy_bpa.ply --n 6000 --procs 8 --surface-points 100000 --out dataset_dummy_suop.npz
```

Samples simulated sonar partials + ground-truth surfaces (coarse 2048 / fine
4096 points) into paired training data. The `_suop` generators resample the
*reconstructed* Route A meshes so the completion prior matches the SUOP target.

**Output:** `dataset.npz`, `dataset_chair.npz`, `dataset_dummy.npz`,
`dataset_dummy_suop.npz`, `dataset_drum_suop.npz` (arrays `parts`, `gts`,
`labels`).
**Time:** ~1–2 min per 6,000-sample dataset (measured: chair 53 s, dummy 86 s,
dummy-suop 116 s).

### 3.2 PCN training — `Route_B_Learned_Completion\02_pcn_training`

```powershell
R:\sonar\Route_B_Learned_Completion\02_pcn_training\train_dummy.bat        # 5 epochs  → pcn_dummy_5ep.pth  (~9 min)
R:\sonar\Route_B_Learned_Completion\02_pcn_training\train_chair.bat        # 150 epochs → pcn_chair.pth     (~4 h)
R:\sonar\Route_B_Learned_Completion\02_pcn_training\train_dummy_suop.bat   # 10 epochs → pcn_dummy_suop_10ep.pth (~28 min)
R:\sonar\Route_B_Learned_Completion\02_pcn_training\train_dummy_suop_epoch_images.bat  # 10 epochs + per-epoch snapshots in checkpoints\
# direct (run from 02_pcn_training so relative --data/--snapshot_dir resolve):
set MODEL_CLASSES=dummy_suop
cd /d R:\sonar\Route_B_Learned_Completion\02_pcn_training
R:\sonar\envs\suop-recon\python.exe train_from_dataset.py --data ..\01_simulation_dataset_generation\dataset_dummy_suop.npz --epochs 10 --bs 64 --snapshot_dir checkpoints\dummy_suop_epoch_images --out pcn_dummy_suop_10ep.pth
```

Coarse-to-fine PCN (encoder-decoder, Chamfer distance loss, Adam lr 1e-3,
StepLR ×0.5 every 40 epochs). `MODEL_CLASSES` env var must match the dataset
(`chair`, `dummy`, `dummy_suop`, `drum_suop`). Supports `--resume` /
`--start_epoch` to continue training.

**Output:**
* `pcn_{chair,dummy,dummy_suop,drum_suop}_*.pth` checkpoints (dict with
  `state_dict`, `classes`, `epoch`)
* `checkpoints\{run}\epoch_XX.pth` per-epoch snapshots (with `--snapshot_dir`)
* `train_*.log` — per-epoch train/val Chamfer + timing

**Time** (RTX 4060 Laptop, bs 64, 6,000 samples):
* 5 epochs ≈ **9 min** (measured 563 s)
* 10 epochs ≈ **28 min** (measured 1,705 s with snapshots)
* 150 epochs ≈ **4.1 h** (measured 14,827 s)
* CPU-only fallback is several× slower.

### 3.3 Inference / completion rendering — `Route_B_Learned_Completion\03_inference_reconstruction`

```powershell
# Complete one held sample with a checkpoint (--checkpoint/--data are required):
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_B_Learned_Completion\03_inference_reconstruction\render_pcn_checkpoint.py --checkpoint R:\sonar\Route_B_Learned_Completion\02_pcn_training\pcn_dummy_10ep.pth --data R:\sonar\Route_B_Learned_Completion\01_simulation_dataset_generation\dataset_dummy.npz --epoch 10 --out R:\sonar\Evaluation_Rendering\comparison_outputs\dummy_epoch_10_comparison.png --sample-index 42 --object-name dummy

# Batch dummy comparisons from the 5- and 10-epoch checkpoints:
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_B_Learned_Completion\03_inference_reconstruction\render_dummy_epoch_comparisons.py

# Contact sheet from per-epoch snapshots:
R:\sonar\envs\suop-recon\python.exe R:\sonar\Route_B_Learned_Completion\03_inference_reconstruction\make_epoch_contact_sheet.py --source R:\sonar\Evaluation_Rendering\comparison_outputs\dummy_suop_epochs --out R:\sonar\Evaluation_Rendering\comparison_outputs\dummy_suop_epochs_contact_sheet.png
```

**Output:**
* `render_pcn_checkpoint.py` → the PNG you pass to `--out` (input partial,
  reference surface, PCN completion side-by-side)
* `render_dummy_epoch_comparisons.py` → `Evaluation_Rendering\comparison_outputs\`
  (`dummy_epoch_05_comparison.png`, `dummy_epoch_10_comparison.png`, …)
* `make_epoch_contact_sheet.py` → `*_contact_sheet.png` in `comparison_outputs\`

**Time:** ~1–5 min per render (GPU inference is fast; CPU is slower).

---

## 4. Evaluation & rendering — `Evaluation_Rendering`

```powershell
# Summary tables of Route A metrics (viewaware + ray-TSDF reconstruction_metrics.json)
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\_summarize_metrics.py

# Summary of registration/meshing manifests
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\_summarize_manifests.py

# Figure: fused cloud vs Poisson vs BPA for chair + dummy
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\render_reconstruction_comparisons.py

# Diagnostics (pose scatter, fused-cloud stats, spacing)
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\check_poses.py
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\diag_fused.py
R:\sonar\envs\suop-recon\python.exe R:\sonar\Evaluation_Rendering\diag_spacing.py
```

**Output:** `Evaluation_Rendering\comparison_outputs\` —
`chair_dummy_reconstruction_comparison.png`, `paper_fig4/5/6.png`,
`dummy_epoch_*_comparison.png`, `*_contact_sheet.png`, `chair_ping0.png`.
**Time:** seconds (summaries/diags) to ~2–3 min (comparison rendering).

---

## 5. End-to-end quickstart (order matters)

```powershell
# 1. Shared preprocessing (≈2 min total)
build_cache.bat → denoise.bat → segment_v3.bat

# 2. Route A backbone (≈1–1.5 h)
multiway_registration_v2.bat → partial_fusion.bat → meshing_pipeline.bat

# 3. Route A variants (≈30–60 min)
reconstruct_viewaware.py chair | tire
reconstruct_suop_tsdf.py both
reconstruct_chair.py --stage all ; reconstruct_dummy.py --stage all
reconstruct_recognizable.py all

# 4. Route B (≈1 h + training)
gen_dummy.bat / gen_chair.bat / gen_dummy_suop.bat
train_dummy.bat                        # quick smoke test, 9 min
train_dummy_suop_epoch_images.bat      # full snapshot run, 28 min
render_pcn_checkpoint.py / render_dummy_epoch_comparisons.py

# 5. Evaluation
_summarize_metrics.py ; render_reconstruction_comparisons.py
```

---

## 6. Data layout

| Path                              | Contents                                     |
| --------------------------------- | -------------------------------------------- |
| `data\cache`                      | Parsed per-case `.npz` (xyz + intensity)     |
| `data\cache_clean`                | SOR-denoised + voxel-downsampled clouds      |
| `data\cache_segmented`            | DBSCAN object segments (v1)                  |
| `data\cache_segmented_v3`         | Plane-removed segments used by Route A (v3)  |
| `data\cache_fused` / `_v2` / `_partial` | Pose-graph fused object clouds (5 each) |
| `data\case_paths_120.txt`         | The 120-case scan list                       |
| `data\*.tsv/.csv/.txt`            | Case lists, LFS inventories, ping lists      |
| `suop_repo\SUOP_dataset`          | Official dataset (git-lfs)                   |
| `suop_detection_reference\`       | SUOP Object-Detection bbox labels            |
| `envs\suop-recon\`                | Conda Python 3.10 environment                |

## 7. Notes

* **Times** were measured on an RTX 4060 Laptop GPU / 8-core CPU from the
  project's own logs; expect longer runs on CPU-only machines (roughly 2–4×).
* All scripts resolve paths relative to the project root, so they can be run
  from any folder — the `.bat` runners simply `cd /d "%~dp0"` for tidy logs.
* Route A needs `data\cache_segmented_v3`; Route B needs Route A meshes for the
  `_suop` datasets and checkpoints for inference.
* Historical run logs (`*.log`) and `*_done.marker` files are kept alongside
  each stage as evidence of completed runs.
