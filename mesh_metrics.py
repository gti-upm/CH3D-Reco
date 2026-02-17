"""
mesh_distance_evaluation.py
----------------------------
Evaluates the geometric accuracy of reconstructed 3D meshes by computing
Hausdorff and L2 (RMS) distances against a reference (ground-truth) model.

Results are saved to an Excel file for further analysis.

Dependencies:
    - pymeshlab
    - pandas
    - numpy
    - opencv-python (cv2)

Usage:
    python mesh_distance_evaluation.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pymeshlab


# ---------------------------------------------------------------------------
# Constants / Configuration
# ---------------------------------------------------------------------------

# Name of the model being evaluated (used for folder path construction)
MODEL_NAME = "kriegerdenkmal"

# Path to the ground-truth (reference) mesh
ORIGINAL_MODEL_PATH = (
    "/3dmodels/FINAL/{MODEL_NAME}/source/Kriegerdenkmal_C/Kriegerdenkmal_C.obj"
)

# Output path for the results spreadsheet
EXCEL_OUTPUT_PATH = (
    f"/3dmodels/FINAL/{MODEL_NAME}/"
    "L2_Haussdorf_distance_metrics.xlsx"
)

# Base directory containing reconstructed model folders
RECONSTRUCTED_MODELS_BASE = (
    f"/3dmodels/FINAL/ReconstructedModels/{MODEL_NAME}"
)

# Experimental parameters to sweep over
CAPTURE_COUNTS = ["50", "125", "250", "500"]   # Number of input captures
TEXTURIZER_VARIANTS = [                         # Triangle count variants (in thousands)
    "Texturizer5000k",
    "Texturizer25000k",
    "Texturizer50000k",
    "Texturizer100000k",
]
MEGAPIXEL_VARIANTS = [1]                        # Megapixel settings (extend as needed)

# Number of samples used for Hausdorff distance computation
HAUSDORFF_SAMPLE_COUNT = 500000


# ---------------------------------------------------------------------------
# Geometry Utilities
# ---------------------------------------------------------------------------

def rescale_meshes_to_unit_diagonal(
    ms: pymeshlab.MeshSet,
    mesh_id_main: int,
    mesh_id_second: int,
) -> None:
    """
    Rescales both meshes uniformly so that the bounding-box diagonal of the
    *main* mesh equals 1. The same scale factor is applied to the second mesh
    to preserve relative proportions between the two.

    Args:
        ms:             Active MeshLab MeshSet containing both meshes.
        mesh_id_main:   ID of the reference (main) mesh used to derive the scale.
        mesh_id_second: ID of the mesh to be compared.

    Raises:
        ValueError: If the main mesh has a zero-length bounding-box diagonal.
    """
    ms.set_current_mesh(mesh_id_main)

    diagonal = ms.current_mesh().bounding_box().diagonal()
    print(f"  Bounding-box diagonal (main mesh): {diagonal:.6f}")

    if diagonal == 0:
        raise ValueError(
            f"Mesh '{mesh_id_main}' has a zero bounding-box diagonal. "
            "Cannot rescale."
        )

    scale_factor = 1.0 / diagonal

    # Apply uniform scale to the main mesh
    ms.compute_matrix_from_translation_rotation_scale(
        scalex=scale_factor,
        scaley=scale_factor,
        scalez=scale_factor,
    )

    # Apply the same scale to the comparison mesh
    ms.set_current_mesh(mesh_id_second)
    ms.compute_matrix_from_translation_rotation_scale(
        scalex=scale_factor,
        scaley=scale_factor,
        scalez=scale_factor,
    )


def compute_hausdorff_distance(
    model_main_path: str,
    model_second_path: str,
    rescale_to_unit_diagonal: bool = True,
) -> tuple[float, float]:
    """
    Computes the one-sided Hausdorff distance and L2 (RMS) distance between
    two triangle meshes using MeshLab's sampling-based approach.

    The distance is measured *from* the main mesh *to* the second mesh, i.e.,
    the main mesh acts as the source of sample points.

    Args:
        model_main_path:          Path to the primary (reconstructed) mesh file.
        model_second_path:        Path to the target (reference/ground-truth) mesh file.
        rescale_to_unit_diagonal: If True, both meshes are rescaled so that the
                                  main mesh bounding-box diagonal equals 1 before
                                  distance computation (makes results scale-invariant).

    Returns:
        A tuple (hausdorff_max, l2_rms) where:
            hausdorff_max - Maximum (one-sided) Hausdorff distance.
            l2_rms        - Root-mean-square (L2) distance over all samples.
    """
    ms = pymeshlab.MeshSet()

    # Load both meshes and record their IDs
    ms.load_new_mesh(model_main_path)
    mesh_id_main = ms.current_mesh_id()

    ms.load_new_mesh(model_second_path)
    mesh_id_second = ms.current_mesh_id()

    # Optionally normalise scale before measuring distances
    if rescale_to_unit_diagonal:
        rescale_meshes_to_unit_diagonal(ms, mesh_id_main, mesh_id_second)

    # Run Hausdorff distance computation (sampled from main mesh to target)
    ms.set_current_mesh(mesh_id_main)
    result = ms.get_hausdorff_distance(
        sampledmesh=mesh_id_main,
        targetmesh=mesh_id_second,
        samplevert=True,
        sampleface=True,
        sampleedge=True,
        samplenum=HAUSDORFF_SAMPLE_COUNT,
    )

    hausdorff_max = result["max"]
    l2_rms = result["RMS"]

    print(f"  Hausdorff (max): {hausdorff_max:.6f}  |  L2 (RMS): {l2_rms:.6f}")

    # Release MeshLab resources
    ms.clear()

    return hausdorff_max, l2_rms


# ---------------------------------------------------------------------------
# Result Helpers
# ---------------------------------------------------------------------------

def build_result_row(
    model: str,
    captures: str,
    triangles_variant: str,
    megapixels: int,
    l2: float,
    hausdorff: float,
) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame with the evaluation metrics for one
    experimental configuration.

    Args:
        model:             Model name identifier.
        captures:          Number of input captures (as string label).
        triangles_variant: Texturizer/triangle-count variant label.
        megapixels:        Megapixel setting used for reconstruction.
        l2:                L2 (RMS) distance (unit-diagonal normalised).
        hausdorff:         Hausdorff max distance (unit-diagonal normalised).

    Returns:
        A single-row pandas DataFrame.
    """
    return pd.DataFrame({
        "modelo":                              [model],
        "n_capturas":                          [captures],
        "triangles(k)":                        [triangles_variant],
        "MP":                                  [str(megapixels)],
        "L2_rescale_to_unit_diagonal":         [l2],
        "Haousdorff_rescale_to_unit_diagonal": [hausdorff],
    })


# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """
    Iterates over all combinations of capture count, texturizer variant, and
    megapixel setting, computes Hausdorff and L2 distances for each
    reconstructed mesh against the reference model, and writes the aggregated
    results to an Excel file.
    """
    results = pd.DataFrame(columns=[
        "modelo",
        "n_capturas",
        "triangles(k)",
        "MP",
        "L2_rescale_to_unit_diagonal",
        "Haousdorff_rescale_to_unit_diagonal",
    ])

    total_configs = len(CAPTURE_COUNTS) * len(TEXTURIZER_VARIANTS) * len(MEGAPIXEL_VARIANTS)
    processed = 0

    for captures in CAPTURE_COUNTS:
        for triangles_variant in TEXTURIZER_VARIANTS:
            for megapixels in MEGAPIXEL_VARIANTS:
                processed += 1

                # Build path to the reconstructed mesh for this configuration
                reconstruction_folder = os.path.join(
                    RECONSTRUCTED_MODELS_BASE,
                    f"capturas{captures}",
                    triangles_variant,
                    str(megapixels),
                )
                reconstructed_mesh_path = os.path.join(
                    reconstruction_folder,
                    "textured_mesh_final_modificado.obj",
                )

                print(
                    f"\n[{processed}/{total_configs}] Evaluating: "
                    f"captures={captures}, triangles={triangles_variant}, MP={megapixels}"
                )
                print(f"  Mesh: {reconstructed_mesh_path}")

                # Skip missing meshes gracefully instead of crashing the whole run
                if not os.path.isfile(reconstructed_mesh_path):
                    print(f"  WARNING: Mesh file not found — skipping.")
                    continue

                # Compute distances (unit-diagonal normalised)
                hausdorff, l2 = compute_hausdorff_distance(
                    reconstructed_mesh_path,
                    ORIGINAL_MODEL_PATH,
                    rescale_to_unit_diagonal=True,
                )

                # Append result row to the results table
                row = build_result_row(
                    model=MODEL_NAME,
                    captures=captures,
                    triangles_variant=triangles_variant,
                    megapixels=megapixels,
                    l2=l2,
                    hausdorff=hausdorff,
                )
                results = pd.concat([results, row], ignore_index=True)

    # Save aggregated results to Excel
    os.makedirs(os.path.dirname(EXCEL_OUTPUT_PATH), exist_ok=True)
    results.to_excel(EXCEL_OUTPUT_PATH, index=False)
    print(f"\nEvaluation complete. Results saved to:\n  {EXCEL_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()