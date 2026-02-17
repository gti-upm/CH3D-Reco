"""
Image Quality Metrics Evaluation Pipeline
==========================================
Computes full-reference, no-reference, and distribution-level image quality
metrics between rendered 3D model captures and original reference images.

Per-image metrics (saved to Excel, one row per image):
    Full-reference : MAE, MSE, PSNR, SSIM, LPIPS (AlexNet)
    No-reference   : BRISQUE, NIQE
    Geometry       : IoU, pixel count difference

Distribution-level metric (single scalar per configuration):
    FID (Fréchet Inception Distance) — measures how similar the *distribution*
    of rendered images is to the distribution of real images. Lower is better.

Author: [Your Name]
Date:   [Date]
"""

import os
from glob import glob

import cv2
import numpy as np
import torch
import pandas as pd
import pyiqa
import lpips
from brisque import BRISQUE
from skimage.metrics import structural_similarity as ssim_metric
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preload LPIPS model once at startup to avoid repeated initialisation overhead
LPIPS_MODEL = lpips.LPIPS(net="alex").to(DEVICE)

# FID resize dimensions — InceptionV3 works best on reasonably sized inputs;
# resizing avoids OOM errors when working with high-resolution captures.
FID_IMAGE_HEIGHT = 2160 // 2   # 1080 px
FID_IMAGE_WIDTH  = 3840 // 2   # 1920 px

# Dataset and experiment parameters
MODEL_NAME    = "kriegerdenkmal"
N_CAPTURES    = ["50", "125", "250", "500"]
TEXTURIZERS   = ["Texturizer5000k", "Texturizer25000k", "Texturizer50000k", "Texturizer100000k"]
MP_VALUES     = [1, 4, 16, 32]

# File system paths
DATASET_ROOT     = "/3dmodels/FINAL"
RESULTS_ROOT     = "/3dmodels/FINAL/ReconstructedModels"
ORIGINAL_IMA_DIR = os.path.join(DATASET_ROOT, MODEL_NAME, "ima_1000")
ORIGINAL_MSK_DIR = os.path.join(DATASET_ROOT, MODEL_NAME, "mask_1000")

# DataFrame column schema for per-image metrics
METRIC_COLUMNS = [
    "Image", "Dif_N_Pixels", "IoU",
    "MAE", "MSE", "SSIM", "PSNR",
    "LPIPS_Alex", "BRISQUE", "NIQE",
]

# ---------------------------------------------------------------------------
# Image I/O Utilities
# ---------------------------------------------------------------------------

def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from disk and convert it to RGB (or grayscale).

    Args:
        path:      Absolute or relative path to the image file.
        grayscale: If True, returns a single-channel grayscale image.

    Returns:
        Image array in RGB (H x W x 3, uint8) or grayscale (H x W, uint8).

    Raises:
        FileNotFoundError: If the image cannot be read from ``path``.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if grayscale:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_binary_mask(path: str, threshold: int = 128) -> np.ndarray:
    """
    Load a mask image and binarise it at ``threshold``.

    Args:
        path:      Path to the mask image.
        threshold: Pixel value above which a pixel is considered foreground.

    Returns:
        Boolean array of shape (H, W).
    """
    gray = load_image(path, grayscale=True)
    return gray > threshold


def load_images_as_tensors(
    rendered_dir: str,
    original_dir: str,
    height: int,
    width: int,
    max_images: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load matched rendered / original image pairs, resize them, and stack
    them into uint8 tensors suitable for FID computation.

    Only filenames present in **both** folders are processed so that the
    two batches are always the same length and in the same order.

    Args:
        rendered_dir: Directory containing rendered PNG images.
        original_dir: Directory containing original reference PNG images.
        height:       Target height after resizing.
        width:        Target width after resizing.
        max_images:   If set, only the first ``max_images`` matched pairs
                      are loaded (useful for quick smoke tests).

    Returns:
        Tuple ``(rendered_tensors, original_tensors)`` where each element
        is a uint8 tensor of shape (N, 3, H, W) on CPU.

    Raises:
        ValueError: If no matching image pairs are found.
    """
    filenames = sorted(os.path.basename(p)
                       for p in glob(os.path.join(rendered_dir, "*.png")))
    if max_images is not None:
        filenames = filenames[:max_images]

    rendered_list: list[torch.Tensor] = []
    original_list: list[torch.Tensor] = []

    for fname in tqdm(filenames, desc="  Loading images for FID", unit="img"):
        rendered_path = os.path.join(rendered_dir, fname)
        original_path = os.path.join(original_dir, fname)

        rendered_img = cv2.imread(rendered_path)
        original_img = cv2.imread(original_path)

        # Skip pairs where either file is missing or unreadable
        if rendered_img is None or original_img is None:
            print(f"  [WARNING] Skipping {fname}: file missing or unreadable.")
            continue

        rendered_img = cv2.resize(rendered_img, (width, height))
        original_img = cv2.resize(original_img, (width, height))

        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # (H, W, C) -> (C, H, W), kept on CPU for FID batch accumulation
        rendered_list.append(torch.from_numpy(rendered_img).permute(2, 0, 1))
        original_list.append(torch.from_numpy(original_img).permute(2, 0, 1))

    if not rendered_list:
        raise ValueError(f"No valid image pairs found in:\n"
                         f"  rendered: {rendered_dir}\n"
                         f"  original: {original_dir}")

    return torch.stack(rendered_list), torch.stack(original_list)

# ---------------------------------------------------------------------------
# Full-Reference Metrics
# ---------------------------------------------------------------------------

def compute_mae(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE) restricted to the masked region.

    The denominator is the number of foreground **pixels** (not pixel x channel),
    so the result represents the average absolute error per spatial location,
    summed across RGB channels. This matches the original definition used in
    prior experiments and ensures comparability with previously reported values.

    Args:
        im1:  First image  (H x W x 3, uint8).
        im2:  Second image (H x W x 3, uint8).
        mask: Boolean mask (H x W); True = foreground pixel.

    Returns:
        Scalar MAE value (sum of channel errors divided by number of pixels).
    """
    mask_f = mask.astype(np.float32)
    im1_f  = im1.astype(np.float32)
    im2_f  = im2.astype(np.float32)

    for c in range(3):
        im1_f[:, :, c] *= mask_f
        im2_f[:, :, c] *= mask_f

    n_pixels = np.sum(mask)  # divide by pixels only, not pixels x channels
    return float(np.sum(np.abs(im1_f - im2_f)) / n_pixels) if n_pixels > 0 else 0.0


def compute_mse(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) restricted to the masked region.

    As with :func:`compute_mae`, the denominator is the number of foreground
    pixels (not pixel x channel) to preserve consistency with prior results.
    Supports both single-channel and multi-channel images.

    Args:
        im1:  First image  (H x W [x C], uint8).
        im2:  Second image (H x W [x C], uint8).
        mask: Boolean mask (H x W).

    Returns:
        Scalar MSE value.
    """
    mask_f = mask.astype(np.float32)
    im1_f  = im1.astype(np.float32)
    im2_f  = im2.astype(np.float32)

    if im1_f.ndim == 3:
        for c in range(im1_f.shape[2]):
            im1_f[:, :, c] *= mask_f
            im2_f[:, :, c] *= mask_f
    else:
        im1_f *= mask_f
        im2_f *= mask_f

    n_pixels = np.sum(mask)  # divide by pixels only, not pixels x channels
    return float(np.sum((im1_f - im2_f) ** 2) / n_pixels) if n_pixels > 0 else 0.0


def compute_psnr(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray,
                 max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) averaged over RGB channels.

    A per-channel MSE is computed and converted to dB. Channels with zero
    MSE (perfect match) contribute ``inf`` to the mean.

    Args:
        im1:       First image  (H x W x 3, uint8).
        im2:       Second image (H x W x 3, uint8).
        mask:      Boolean mask (H x W).
        max_pixel: Dynamic range of the images (default 255.0 for uint8).

    Returns:
        Mean PSNR in dB across channels.
    """
    channel_psnr = []
    for c in range(3):
        mse = compute_mse(im1[:, :, c], im2[:, :, c], mask)
        if mse > 0:
            channel_psnr.append(20.0 * np.log10(max_pixel / np.sqrt(mse)))
        else:
            channel_psnr.append(float("inf"))
    return float(np.mean(channel_psnr))


def compute_ssim(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) computed on the luminance channel
    within the masked region.

    Args:
        im1:  First image  (H x W x 3, uint8), RGB.
        im2:  Second image (H x W x 3, uint8), RGB.
        mask: Boolean mask (H x W).

    Returns:
        Mean SSIM score over foreground pixels.
    """
    mask_f = mask.astype(np.float32)
    gray1  = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY).astype(np.float32) * mask_f
    gray2  = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY).astype(np.float32) * mask_f

    data_range = gray2.max() - gray2.min()
    _, ssim_map = ssim_metric(gray1, gray2, data_range=data_range, full=True)

    return float(np.mean(ssim_map[mask]))


def compute_lpips(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS) with AlexNet backbone,
    evaluated only within the masked region.

    Images are normalised to [-1, 1] as required by the LPIPS network.

    Args:
        im1:  First image  (H x W x 3, uint8), RGB.
        im2:  Second image (H x W x 3, uint8), RGB.
        mask: Boolean mask (H x W).

    Returns:
        Scalar LPIPS distance (lower = more similar).
    """
    mask_f    = mask.astype(np.float32)
    to_tensor = lambda img: (
        torch.tensor(img.astype(np.float32) / 127.5 - 1.0)
        .permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    )
    t1     = to_tensor(im1)
    t2     = to_tensor(im2)
    mask_t = torch.tensor(mask_f).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        score = LPIPS_MODEL(t1 * mask_t, t2 * mask_t)

    return float(score.item())

# ---------------------------------------------------------------------------
# No-Reference Metrics
# ---------------------------------------------------------------------------

def compute_brisque(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) score.

    The mask is applied before computing the score so that background
    regions do not contaminate the natural scene statistics.

    Args:
        image: Input image (H x W x 3, uint8), RGB.
        mask:  Boolean mask (H x W).

    Returns:
        BRISQUE score (lower = better perceptual quality).
    """
    mask_f     = mask.astype(np.float32)
    img_masked = image.copy()
    for c in range(3):
        img_masked[:, :, c] = img_masked[:, :, c] * mask_f

    return float(BRISQUE(url=False).score(img=img_masked))


def compute_niqe(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Natural Image Quality Evaluator (NIQE) score.

    The image is normalised to [0, 1] and the mask is applied before
    passing it to the pyiqa framework.

    Args:
        image: Input image (H x W x 3, uint8), RGB.
        mask:  Boolean mask (H x W).

    Returns:
        NIQE score (lower = better perceptual quality).
    """
    iqa_metric = pyiqa.create_metric("niqe", device=DEVICE)

    img_t  = (torch.tensor(image.astype(np.float32) / 255.0)
              .permute(2, 0, 1).unsqueeze(0).to(DEVICE))
    mask_t = (torch.tensor(mask.astype(np.float32))
              .unsqueeze(0).unsqueeze(0).to(DEVICE))

    with torch.no_grad():
        score = iqa_metric(img_t * mask_t)

    return float(score.item())

# ---------------------------------------------------------------------------
# Distribution-Level Metric: FID
# ---------------------------------------------------------------------------

def compute_fid(rendered_dir: str, original_dir: str) -> float:
    """
    Compute the Frechet Inception Distance (FID) between all rendered images
    in ``rendered_dir`` and all reference images in ``original_dir``.

    FID measures the distance between the Gaussian distributions fitted to
    InceptionV3 feature vectors of both image sets. Lower values indicate
    that the rendered distribution is closer to the reference distribution.

    Unlike per-image metrics, FID is a **dataset-level** score: it requires
    a sufficient number of images (ideally >= 50) to be statistically reliable.

    Args:
        rendered_dir: Directory containing rendered PNG images.
        original_dir: Directory containing original reference PNG images.

    Returns:
        Scalar FID score.
    """
    print("  [FID] Loading image pairs...")
    rendered_tensors, original_tensors = load_images_as_tensors(
        rendered_dir, original_dir,
        height=FID_IMAGE_HEIGHT,
        width=FID_IMAGE_WIDTH,
    )
    print(f"  [FID] Loaded {rendered_tensors.shape[0]} image pairs.")

    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)

    # Move tensors to device only during update to save VRAM
    fid_metric.update(original_tensors.to(DEVICE), real=True)
    fid_metric.update(rendered_tensors.to(DEVICE), real=False)

    return float(fid_metric.compute().item())

# ---------------------------------------------------------------------------
# Geometry Metrics
# ---------------------------------------------------------------------------

def compute_mask_geometry(rendered_mask: np.ndarray,
                          original_mask: np.ndarray) -> tuple[float, float]:
    """
    Compute Intersection-over-Union (IoU) and absolute pixel-count difference
    between the rendered and original masks.

    Args:
        rendered_mask: Boolean mask from the rendered image (H x W).
        original_mask: Boolean mask from the original reference (H x W).

    Returns:
        Tuple ``(iou, pixel_difference)`` where:
            iou              - ratio of intersection to union area [0, 1].
            pixel_difference - |rendered_pixels - original_pixels|.
    """
    intersection   = np.bitwise_and(rendered_mask, original_mask)
    union          = np.bitwise_or(rendered_mask,  original_mask)
    n_intersection = int(np.sum(intersection))
    n_union        = int(np.sum(union))
    pixel_diff     = abs(int(np.sum(rendered_mask)) - int(np.sum(original_mask)))

    iou = n_intersection / n_union if n_union > 0 else 0.0
    return iou, pixel_diff

# ---------------------------------------------------------------------------
# Per-Image Evaluation
# ---------------------------------------------------------------------------

def evaluate_image_pair(
    image_name:       str,
    rendered_ima_dir: str,
    original_ima_dir: str,
    rendered_msk_dir: str,
    original_msk_dir: str,
) -> dict | None:
    """
    Load a single rendered / original image pair, compute all per-image
    metrics, and return the results as a flat dictionary.

    Args:
        image_name:       Filename (e.g. ``"frame_0001.png"``).
        rendered_ima_dir: Directory of rendered images.
        original_ima_dir: Directory of original reference images.
        rendered_msk_dir: Directory of rendered masks.
        original_msk_dir: Directory of original reference masks.

    Returns:
        Dictionary mapping each metric column name to its value, or
        ``None`` if any file is missing or an error occurs.
    """
    try:
        rendered = load_image(os.path.join(rendered_ima_dir, image_name))
        original = load_image(os.path.join(original_ima_dir, image_name))

        rendered_mask = load_binary_mask(os.path.join(rendered_msk_dir, image_name))
        original_mask = load_binary_mask(os.path.join(original_msk_dir, image_name))

        union_mask = np.bitwise_or(rendered_mask, original_mask)

        # --- Geometry ---
        iou, dif_pixels = compute_mask_geometry(rendered_mask, original_mask)

        # --- Full-reference (restricted to the union region for fair comparison) ---
        rendered_roi = np.zeros_like(rendered)
        original_roi = np.zeros_like(original)
        rendered_roi[union_mask] = rendered[union_mask]
        original_roi[union_mask] = original[union_mask]

        mae   = compute_mae(rendered_roi, original_roi, union_mask)
        mse   = compute_mse(rendered_roi, original_roi, union_mask)
        psnr  = compute_psnr(rendered_roi, original_roi, union_mask)
        s_sim = compute_ssim(rendered, original, union_mask)
        lp    = compute_lpips(rendered, original, union_mask)

        # --- No-reference (evaluated on the rendered image only) ---
        brisque = compute_brisque(rendered, rendered_mask)
        niqe    = compute_niqe(rendered,    rendered_mask)

    except Exception as exc:
        print(f"  [WARNING] Skipping {image_name}: {exc}")
        return None

    return {
        "Image":        image_name,
        "Dif_N_Pixels": dif_pixels,
        "IoU":          iou,
        "MAE":          mae,
        "MSE":          mse,
        "SSIM":         s_sim,
        "PSNR":         psnr,
        "LPIPS_Alex":   lp,
        "BRISQUE":      brisque,
        "NIQE":         niqe,
    }


def build_results_dataframe(rows: list[dict], fid_score: float) -> pd.DataFrame:
    """
    Convert a list of per-image metric dictionaries into a DataFrame, append
    a summary row with column-wise means, and include the FID score.

    FID is a distribution-level metric so it has no meaningful per-image
    value; it is recorded only in the ``"TOTAL"`` summary row. Per-image
    rows will have NaN in the FID column.

    Args:
        rows:      List of dicts produced by :func:`evaluate_image_pair`.
        fid_score: FID score for the whole configuration.

    Returns:
        DataFrame with one row per image plus a final ``"TOTAL"`` row.
    """
    all_columns = METRIC_COLUMNS + ["FID"]
    df          = pd.DataFrame(rows, columns=METRIC_COLUMNS)

    # Build summary row: mean of numeric columns + FID (dataset-level only)
    numeric_cols     = METRIC_COLUMNS[1:]
    summary          = df[numeric_cols].mean().to_dict()
    summary["Image"] = "TOTAL"
    summary["FID"]   = fid_score

    # Per-image rows: FID is not applicable at image level
    df["FID"] = float("nan")

    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    return df[all_columns]

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Iterate over all experiment configurations and, for each combination
    of capture count, texturizer setting, and megapixel value:

    1. Compute per-image quality metrics (MAE, MSE, PSNR, SSIM, LPIPS,
       BRISQUE, NIQE, IoU, pixel difference).
    2. Compute the FID score for the whole rendered set vs. the reference set.
    3. Save everything to an Excel file (one row per image + a TOTAL row
       that also contains the FID score).

    Already-processed configurations (Excel file already exists) are skipped.
    """
    print(f"Using device: {DEVICE}\n")

    for captures in N_CAPTURES:
        for texturizer in TEXTURIZERS:
            for mp in MP_VALUES:

                # --- Resolve paths for this configuration ---
                excel_path = os.path.join(
                    RESULTS_ROOT, MODEL_NAME, "union", "same",
                    f"captures{captures}_{texturizer}_{mp}"
                    f"_image_metrics_union.xlsx"
                )

                if os.path.exists(excel_path):
                    print(f"[SKIP] Already processed: {excel_path}")
                    continue

                sfm_folder       = (f"capturas{captures}")
                main_folder      = os.path.join(RESULTS_ROOT, MODEL_NAME,
                                                sfm_folder, texturizer, str(mp))
                rendered_ima_dir = os.path.join(main_folder, "captures300")
                rendered_msk_dir = os.path.join(main_folder, "masks300")

                if not os.path.isdir(rendered_ima_dir):
                    print(f"[WARNING] Directory not found, skipping: {rendered_ima_dir}")
                    continue

                image_files = sorted(
                    f for f in os.listdir(rendered_ima_dir) if f.endswith(".png")
                )
                if not image_files:
                    print(f"[WARNING] No PNG images found in: {rendered_ima_dir}")
                    continue

                print(f"\n{'='*60}")
                print(f"[INFO] captures={captures} | texturizer={texturizer} "
                      f"| mp={mp} | images={len(image_files)}")
                print(f"{'='*60}")

                # --- Step 1: Per-image metrics ---
                rows = []
                for fname in tqdm(image_files, desc="  Per-image metrics", unit="img"):
                    result = evaluate_image_pair(
                        fname,
                        rendered_ima_dir, ORIGINAL_IMA_DIR,
                        rendered_msk_dir, ORIGINAL_MSK_DIR,
                    )
                    if result is not None:
                        rows.append(result)

                if not rows:
                    print("[WARNING] No valid results; skipping Excel export.")
                    continue

                # --- Step 2: FID (distribution-level, computed once per config) ---
                print("\n  Computing FID score...")
                try:
                    fid_score = compute_fid(rendered_ima_dir, ORIGINAL_IMA_DIR)
                    print(f"  [FID] Score: {fid_score:.4f}")
                except Exception as exc:
                    print(f"  [WARNING] FID computation failed: {exc}")
                    fid_score = float("nan")

                # --- Step 3: Export results to Excel ---
                df = build_results_dataframe(rows, fid_score)
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                df.to_excel(excel_path, index=False)
                print(f"\n[OK] Saved: {excel_path}")


if __name__ == "__main__":
    main()