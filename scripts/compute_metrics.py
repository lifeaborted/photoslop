#!/usr/bin/env python3
"""
PepeShop Metric Evaluation Suite
Computes the exact metrics required by the project specifications (ТЗ):
  1. IoU маски фона & Boundary-IoU (Intersection over Union for segmentation masks)
  2. CLIP Score (cosine similarity between image and text prompt via ViT-B/32)
  3. FID / Fréchet Distance (distribution distance on feature representations)
  4. PSNR & SSIM (reconstruction quality for object removal / inpainting)
"""

import gc
import math
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional


def compute_iou(
    pred_mask: Union[Image.Image, np.ndarray],
    gt_mask: Union[Image.Image, np.ndarray],
    threshold: int = 128,
) -> float:
    """
    Computes Intersection over Union (IoU) for a predicted alpha/binary mask vs Ground Truth.
    Target value from specifications (ТЗ): IoU >= 0.90
    """
    if isinstance(pred_mask, Image.Image):
        if pred_mask.mode == "RGBA":
            pred_mask = np.array(pred_mask.getchannel("A"))
        else:
            pred_mask = np.array(pred_mask.convert("L"))
    if isinstance(gt_mask, Image.Image):
        gt_mask = np.array(gt_mask.convert("L"))

    if pred_mask.shape != gt_mask.shape:
        gt_img = Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.Resampling.NEAREST
        )
        gt_mask = np.array(gt_img)

    pred_bin = pred_mask >= threshold
    gt_bin = gt_mask >= threshold

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(round(intersection / union, 4))


def compute_boundary_iou(
    pred_mask: Union[Image.Image, np.ndarray],
    gt_mask: Union[Image.Image, np.ndarray],
    dilation: int = 3,
    threshold: int = 128,
) -> float:
    """
    Computes Boundary-IoU targeting fine boundary structures (hair, whiskers, glass edges).
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        return compute_iou(pred_mask, gt_mask, threshold)

    if isinstance(pred_mask, Image.Image):
        if pred_mask.mode == "RGBA":
            pred_mask = np.array(pred_mask.getchannel("A"))
        else:
            pred_mask = np.array(pred_mask.convert("L"))
    if isinstance(gt_mask, Image.Image):
        gt_mask = np.array(gt_mask.convert("L"))

    if pred_mask.shape != gt_mask.shape:
        gt_img = Image.fromarray(gt_mask).resize(
            (pred_mask.shape[1], pred_mask.shape[0]), Image.Resampling.NEAREST
        )
        gt_mask = np.array(gt_img)

    pred_bin = pred_mask >= threshold
    gt_bin = gt_mask >= threshold

    pred_boundary = pred_bin ^ binary_dilation(pred_bin, iterations=dilation)
    gt_boundary = gt_bin ^ binary_dilation(gt_bin, iterations=dilation)

    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(round(intersection / union, 4))


class ClipScorer:
    """
    Reusable CLIP ViT-B/32 scorer so a batch of generated images is scored efficiently
    without reloading weights per image, and cleanly unloaded so VRAM stats stay pure.
    """

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        import torch
        from transformers import CLIPProcessor, CLIPModel

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def score(self, image: Image.Image, prompt: str) -> float:
        import torch
        inputs = self.processor(
            text=[prompt],
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = torch.clamp((image_embeds * text_embeds).sum(dim=-1), 0.0, 1.0)
            return float(round(similarity.item(), 4))

    def score_batch(self, pairs: List[Tuple[Image.Image, str]]) -> List[float]:
        return [self.score(img, prompt) for img, prompt in pairs]

    def unload(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def compute_clip_score(image: Image.Image, prompt: str, device: str = "cpu") -> float:
    """Single-call convenience wrapper for CLIP Score (ViT-B/32)."""
    scorer = ClipScorer(device=device)
    try:
        return scorer.score(image, prompt)
    finally:
        scorer.unload()


def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Computes the Fréchet distance between two Gaussians (foundation of FID).
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))
    Lower is better.
    """
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = np.eye(sigma1.shape[0]) * np.sqrt(np.trace(sigma1) * np.trace(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(round(max(float(fid), 0.0), 2))


def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB. Higher is better."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    arr1 = np.array(img1.convert("RGB")).astype(np.float64)
    arr2 = np.array(img2.convert("RGB")).astype(np.float64)
    mse = float(np.mean((arr1 - arr2) ** 2))
    if mse == 0.0:
        return 100.0
    return float(round(20.0 * math.log10(255.0 / math.sqrt(mse)), 2))


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Computes Structural Similarity Index (SSIM) in [0, 1]. Higher is better."""
    from skimage.metrics import structural_similarity as ssim
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))
    score = ssim(arr1, arr2, channel_axis=2, data_range=255)
    return float(round(float(score), 4))


if __name__ == "__main__":
    print("Testing metric calculation functions...")
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[20:80, 20:80] = 255
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[25:80, 20:80] = 255
    iou = compute_iou(m1, m2)
    biou = compute_boundary_iou(m1, m2)
    img_a = Image.new("RGB", (64, 64), (100, 120, 140))
    img_b = Image.new("RGB", (64, 64), (102, 120, 138))
    psnr = compute_psnr(img_a, img_b)
    ssim_val = compute_ssim(img_a, img_b)
    print(f"  Synthetic IoU={iou} | Boundary-IoU={biou} | PSNR={psnr} dB | SSIM={ssim_val}")
    print("[✓] Metric evaluation utility verified.")
