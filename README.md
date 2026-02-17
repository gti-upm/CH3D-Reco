# CH3D-Reco Evaluation Toolkit
A set of Python scripts for **quantitatively evaluating the accuracy of photogrammetry-based 3D mesh reconstructions**. The pipeline covers the full evaluation workflow: aligning reconstructed meshes to a ground-truth reference frame, measuring geometric fidelity with Hausdorff and L2 distances, and assessing visual quality with a comprehensive suite of image metrics.

## About

This repository contains the code required to reproduce and analyse the 3D model reconstruction and quality evaluation experiments presented in the article:

> **DOI:** [https://doi.org/10.1016/j.image.2026.117514](https://doi.org/10.1016/j.image.2026.117514)

The dataset used in these experiments is publicly available on Zenodo:

> **Dataset:** [https://zenodo.org/records/18598772](https://zenodo.org/records/18598772)

If you use this code or dataset in your own work, please cite the article above.
---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Scripts](#scripts)
  - [camera\_based\_mesh\_alignment.py](#1-camera_based_mesh_alignmentpy)
  - [mesh\_metrics.py](#2-mesh_metricspy)
  - [image\_metrics.py](#3-image_metricspy)
- [Installation](#installation)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Dependencies](#dependencies)

---

## Overview

When evaluating photogrammetric reconstruction pipelines, a common challenge is that the reconstructed mesh lives in an **arbitrary coordinate frame** — it needs to be aligned to the ground-truth model before any distance measurement is meaningful. This toolkit addresses the full workflow:

```
Input captures + ground-truth mesh
        │
        ▼
① camera_based_mesh_alignment.py   ← align reconstructed mesh to GT frame
        │
        ▼
② mesh_metrics.py                  ← geometric accuracy (Hausdorff, L2/RMS)
        │
        ▼
③ image_metrics.py                 ← visual quality (SSIM, PSNR, LPIPS, FID, …)
        │
        ▼
Excel reports (.xlsx)
```

---

## Repository Structure

```
.
├── camera_based_mesh_alignment.py   # Step 1 – align mesh via camera correspondences
├── mesh_metrics.py                  # Step 2 – geometric distance metrics
├── image_metrics.py                 # Step 3 – image quality metrics on renders
└── README.md
```

---

## Pipeline

### Step 1 — Mesh Alignment

Reconstructed meshes are defined in an arbitrary SfM (Structure-from-Motion) coordinate system. `camera_based_mesh_alignment.py` brings them into the ground-truth world frame by:

1. Matching camera centers between the SfM reconstruction and the known ground-truth poses.
2. Computing the optimal linear transformation (rotation + scale + translation) via **least-squares fitting**.
3. Applying the transformation, plus a corrective 90° rotation if needed, directly to the OBJ vertex data.

### Step 2 — Geometric Metrics

`mesh_metrics.py` measures how closely the aligned mesh approximates the ground-truth surface using MeshLab's sampling-based **Hausdorff distance** computation. Both meshes are optionally normalised to a unit bounding-box diagonal for scale-invariant comparison.

### Step 3 — Image Quality Metrics

`image_metrics.py` renders the aligned mesh from known camera poses and compares the resulting images against the original reference photographs using a comprehensive set of **full-reference**, **no-reference**, and **distribution-level** metrics.

---

## Scripts

### 1. `camera_based_mesh_alignment.py`

Aligns a reconstructed OBJ mesh to the ground-truth coordinate system using camera center correspondences.

**How it works:**

- Reads per-image camera calibration data (`CameraCalib.txt`, `PictureNames.txt`) produced by the SfM pipeline.
- Reads ground-truth camera extrinsics from per-image JSON metadata files.
- For each image present in both sets, computes the 3D camera center `C = -R⁻¹t`.
- Calls `least_squares_fit(A, B)` to find the linear transform `M` and translation `t` such that `M·B + t ≈ A` in the least-squares sense.
- Writes the transformed mesh to a new OBJ file (`textured_mesh_final_modified.obj`).

### 2. `mesh_metrics.py`

Computes **geometric distance metrics** between the aligned reconstructed mesh and the ground-truth reference mesh.

**Metrics computed:**

| Metric | Description |
|---|---|
| **Hausdorff distance (max)** | Maximum closest-point distance from any sample on the reconstructed mesh to the reference surface. Sensitive to outliers / worst-case errors. |
| **L2 / RMS distance** | Root-mean-square of all sample-to-surface distances. Reflects average geometric deviation. |

Both metrics are reported after normalising both meshes so the reconstructed mesh bounding-box diagonal equals 1, making results **scale-invariant** and comparable across models.


### 3. `image_metrics.py`

Evaluates **visual quality** by comparing rendered images of the reconstructed model against original reference photographs. Metrics are computed per image and aggregated into an Excel report.

**Metrics computed:**

| Category | Metric | Description |
|---|---|---|
| Full-reference | **MAE** | Mean Absolute Error per foreground pixel |
| Full-reference | **MSE** | Mean Squared Error per foreground pixel |
| Full-reference | **PSNR** | Peak Signal-to-Noise Ratio (dB); higher is better |
| Full-reference | **SSIM** | Structural Similarity Index [0–1]; higher is better |
| Full-reference | **LPIPS** | Learned Perceptual Image Patch Similarity (AlexNet); lower is better |
| No-reference | **BRISQUE** | Blind/Referenceless Image Spatial Quality Evaluator; lower is better |
| No-reference | **NIQE** | Natural Image Quality Evaluator; lower is better |
| Geometry | **IoU** | Intersection-over-Union of rendered vs. reference object masks |
| Geometry | **Pixel difference** | Absolute difference in foreground pixel count |
| Distribution | **FID** | Fréchet Inception Distance — distribution similarity of rendered vs. real images; lower is better |

All per-pixel metrics are restricted to the **union mask** of rendered and reference foreground regions for a fair comparison.

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/your-repo.git
cd your-repo

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `pymeshlab` wheels are available for Python 3.8–3.11 on Linux, macOS, and Windows. GPU support (for LPIPS and FID) requires a CUDA-compatible PyTorch installation; the scripts fall back to CPU automatically.


## Configuration

All user-configurable parameters are defined as **constants at the top of each script** — no command-line arguments are required. Edit the relevant section before running:

**Shared parameters (appear in all three scripts):**

| Constant | Description |
|---|---|
| `MODEL_NAME` | Name of the building / scene being evaluated |
| `CAPTURE_COUNTS` | List of input capture subset sizes to evaluate (e.g. `["50", "125", "250", "500"]`) |
| `TEXTURIZER_VARIANTS` | Mesh resolution variants to evaluate |
| `MEGAPIXEL_VARIANTS` | Render resolution settings |

**Path constants (update to match your file system):**

| Constant | Description |
|---|---|
| `DATASET_ROOT` | Root directory of the original dataset |
| `RECONSTRUCTED_MODELS_BASE` | Root directory of all reconstructed model outputs |
| `ORIGINAL_MODEL_PATH` | Path to the ground-truth reference mesh (`.obj`) |
| `EXCEL_OUTPUT_PATH` | Where to save the results spreadsheet |

---

## Output Files

Each script writes its results to an Excel file (`.xlsx`):

| Script | Output file | Content |
|---|---|---|
| `camera_based_mesh_alignment.py` | `textured_mesh_final_modified.obj` (per config) | Aligned OBJ mesh, written alongside the original |
| `mesh_metrics.py` | `L2_Haussdorf_distance_metrics.xlsx` | One row per configuration: Hausdorff max + L2 RMS |
| `image_metrics.py` | `captures{N}_{texturizer}_{mp}_image_metrics_union.xlsx` | One row per image (all metrics) + a final TOTAL row with column means and FID score |

