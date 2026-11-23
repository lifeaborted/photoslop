#!/usr/bin/env python3
"""
PepeShop Reproducible Benchmark Dataset Builder
Downloads a compact, diverse validation set (~15-25 MB) into data/benchmark_dataset/
(gitignored) for objective KT 2 & KT 3 evaluation:
  1. Background Removal (bg_removal/):
     - 15 real images + Ground Truth masks from DUTS-TE (nobg/DUTS, split='test')
     - 15 real images + Ground Truth masks from DIS5K (nobg/DIS5K, split='DIS_VD')
  2. Object Removal & Text-Guided Inpainting (inpainting/):
     - 10 real 512x512 photos + standardized binary masks + 10 inpaint prompts
  3. Text-to-Image Generation (txt2img_prompts.json):
     - 15 diverse standardized prompts covering portraits, products, landscapes, textures

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --bg-count 20
"""

import io
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import requests
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "benchmark_dataset"
BG_IMG_DIR = DATASET_DIR / "bg_removal" / "images"
BG_MASK_DIR = DATASET_DIR / "bg_removal" / "masks"
INP_IMG_DIR = DATASET_DIR / "inpainting" / "images"
INP_MASK_DIR = DATASET_DIR / "inpainting" / "masks"
INP_ERASE_MASK_DIR = DATASET_DIR / "inpainting" / "erase_masks"

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

TXT2IMG_PROMPTS: List[Dict[str, str]] = [
    {"id": "p01", "category": "product", "prompt": "a ripe red apple with water droplets on a rustic wooden table, studio photography, 8k"},
    {"id": "p02", "category": "product", "prompt": "minimalist white ceramic coffee mug on marble countertop, soft morning sunlight, product photo"},
    {"id": "p03", "category": "vehicle", "prompt": "a sleek red sports car parked on wet city asphalt at sunset, reflections, photorealistic"},
    {"id": "p04", "category": "portrait", "prompt": "close-up portrait of an elderly craftsman in a workshop, warm natural window light, sharp focus"},
    {"id": "p05", "category": "animal", "prompt": "a fluffy orange tabby cat sitting on a windowsill looking outside, bokeh background, 85mm lens"},
    {"id": "p06", "category": "landscape", "prompt": "snowy alpine mountain peaks reflected in a calm turquoise glacial lake at dawn, landscape photography"},
    {"id": "p07", "category": "architecture", "prompt": "modern scandinavian living room interior with oak furniture, indoor plants and large windows"},
    {"id": "p08", "category": "cyberpunk", "prompt": "rainy neon-lit cyberpunk street alley in Tokyo at night, glowing signs, puddles, cinematic"},
    {"id": "p09", "category": "food", "prompt": "freshly baked pepperoni pizza with melted cheese and basil leaves on a dark stone board, top view"},
    {"id": "p10", "category": "nature", "prompt": "macro photograph of a blue morpho butterfly resting on a green fern leaf, dew drops"},
    {"id": "p11", "category": "illustration", "prompt": "watercolor botanical illustration of lavender sprigs and wildflowers on textured cotton paper"},
    {"id": "p12", "category": "still_life", "prompt": "vintage brass compass and leather-bound notebook on an antique world map, warm candlelight"},
    {"id": "p13", "category": "urban", "prompt": "historic cobblestone European street cafe with red umbrellas and flower boxes, sunny afternoon"},
    {"id": "p14", "category": "glass", "prompt": "crystal perfume bottle on frosted glass surface with dramatic rim lighting, commercial shot"},
    {"id": "p15", "category": "scifi", "prompt": "astronaut standing on a red rocky Mars canyon looking at Earth in the dark starry sky, cinematic"},
]

INPAINT_PROMPTS: List[str] = [
    "a bright red ceramic vase with fresh tulips, natural lighting, photorealistic",
    "a fluffy golden retriever puppy sitting peacefully, sharp focus, realistic",
    "a vintage brown leather backpack resting naturally, soft shadows",
    "a sleek metallic coffee thermos bottle, studio lighting, realistic reflections",
    "a wooden wicker basket filled with green apples, natural daylight",
    "a small potted succulent plant in a terracotta pot, detailed texture",
    "a classic black retro film camera on the surface, photorealistic",
    "a pair of stylish aviator sunglasses with subtle reflections",
    "a woven straw sun hat with a dark blue ribbon, natural lighting",
    "a small glowing brass lantern casting warm soft light",
]


def _download_image(url: str, timeout: int = 30) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def _fetch_hf_samples(
    dataset: str,
    split: str,
    image_col: str,
    mask_col: str,
    count: int,
    stride: int,
    prefix: str,
    max_side: int = 1024,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    print(f"Fetching {count} samples from {dataset} ({split})...")

    # Query small pages with stride so samples span diverse object classes
    fetched = 0
    page = 0
    while fetched < count and page < count * 3:
        offset = page * stride
        page += 1
        params = {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": 1,
        }
        try:
            r = requests.get(HF_ROWS_URL, params=params, timeout=20)
            r.raise_for_status()
            rows = r.json().get("rows", [])
            if not rows:
                break
            row = rows[0]["row"]
            img_meta = row.get(image_col)
            msk_meta = row.get(mask_col)
            if not isinstance(img_meta, dict) or not isinstance(msk_meta, dict):
                continue

            img = _download_image(img_meta["src"]).convert("RGB")
            msk = _download_image(msk_meta["src"]).convert("L")

            # Cap extreme 4K DIS5K resolutions to max_side while keeping aspect ratio
            if max(img.size) > max_side:
                ratio = max_side / max(img.size)
                new_size = (max(64, int(img.width * ratio)), max(64, int(img.height * ratio)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                msk = msk.resize(new_size, Image.Resampling.NEAREST)
            elif msk.size != img.size:
                msk = msk.resize(img.size, Image.Resampling.NEAREST)

            sample_id = f"{prefix}_{fetched + 1:02d}"
            img_path = BG_IMG_DIR / f"{sample_id}.jpg"
            msk_path = BG_MASK_DIR / f"{sample_id}.png"
            img.save(img_path, format="JPEG", quality=92)
            msk.save(msk_path, format="PNG")

            records.append({
                "id": sample_id,
                "source_dataset": dataset,
                "split": split,
                "row_offset": offset,
                "source_name": row.get("image_name", f"row_{offset}"),
                "width": img.width,
                "height": img.height,
                "image": str(img_path.relative_to(DATASET_DIR)).replace("\\", "/"),
                "mask": str(msk_path.relative_to(DATASET_DIR)).replace("\\", "/"),
            })
            fetched += 1
            print(f"  [{fetched}/{count}] Saved {sample_id} ({img.width}x{img.height})")
        except Exception as e:
            print(f"  [!] Skipped offset={offset} ({e})")
            continue

    return records


def _center_crop_512(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((512, 512), Image.Resampling.LANCZOS)


def _build_inpainting_set(bg_records: List[Dict[str, Any]], count: int = 10) -> List[Dict[str, Any]]:
    """
    Creates 10 standardized 512x512 images + two mask types:
      - inpaint mask (center-offset box/ellipse for text-guided object generation)
      - erase mask (known background/texture patch for objective LaMa PSNR/SSIM reconstruction)
    """
    INP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    INP_MASK_DIR.mkdir(parents=True, exist_ok=True)
    INP_ERASE_MASK_DIR.mkdir(parents=True, exist_ok=True)

    selected = bg_records[:count]
    inp_records: List[Dict[str, Any]] = []

    for idx, rec in enumerate(selected):
        src_img = Image.open(DATASET_DIR / rec["image"]).convert("RGB")
        crop_img = _center_crop_512(src_img)

        sample_id = f"inp_{idx + 1:02d}"
        img_out = INP_IMG_DIR / f"{sample_id}.png"
        crop_img.save(img_out, format="PNG")

        # 1. Text-guided inpaint mask (~160x160 central-lower region)
        inp_mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(inp_mask)
        if idx % 2 == 0:
            draw.rounded_rectangle([168, 180, 344, 356], radius=24, fill=255)
        else:
            draw.ellipse([164, 176, 348, 360], fill=255)
        inp_mask_out = INP_MASK_DIR / f"{sample_id}.png"
        inp_mask.save(inp_mask_out, format="PNG")

        # 2. LaMa object/defect erasure mask (clean 96x96 stroke/patch for objective PSNR/SSIM)
        erase_mask = Image.new("L", (512, 512), 0)
        edraw = ImageDraw.Draw(erase_mask)
        if idx % 3 == 0:
            # Diagonal wire / scratch mask
            edraw.line([(90, 120), (410, 210)], fill=255, width=22)
        elif idx % 3 == 1:
            # Corner watermark / stamp box
            edraw.rectangle([330, 390, 470, 465], fill=255)
        else:
            # Irregular object blotch
            edraw.ellipse([200, 210, 312, 322], fill=255)
        erase_mask_out = INP_ERASE_MASK_DIR / f"{sample_id}.png"
        erase_mask.save(erase_mask_out, format="PNG")

        inp_records.append({
            "id": sample_id,
            "source_bg_id": rec["id"],
            "image": str(img_out.relative_to(DATASET_DIR)).replace("\\", "/"),
            "inpaint_mask": str(inp_mask_out.relative_to(DATASET_DIR)).replace("\\", "/"),
            "erase_mask": str(erase_mask_out.relative_to(DATASET_DIR)).replace("\\", "/"),
            "prompt": INPAINT_PROMPTS[idx % len(INPAINT_PROMPTS)],
        })

    print(f"[✓] Built {len(inp_records)} standardized 512x512 inpainting & erasure test pairs")
    return inp_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact reproducible benchmark dataset")
    parser.add_argument("--duts-count", type=int, default=15, help="Number of DUTS-TE images (default: 15)")
    parser.add_argument("--dis-count", type=int, default=15, help="Number of DIS5K images (default: 15)")
    parser.add_argument("--inpaint-count", type=int, default=10, help="Number of inpainting/erasure pairs (default: 10)")
    args = parser.parse_args()

    BG_IMG_DIR.mkdir(parents=True, exist_ok=True)
    BG_MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("         PepeShop Benchmark Dataset Builder (DUTS-TE + DIS5K)             ")
    print("==========================================================================")

    duts_recs = _fetch_hf_samples(
        dataset="nobg/DUTS",
        split="test",
        image_col="image",
        mask_col="mask",
        count=args.duts_count,
        stride=280,
        prefix="duts",
    )
    dis_recs = _fetch_hf_samples(
        dataset="nobg/DIS5K",
        split="DIS_VD",
        image_col="image",
        mask_col="label",
        count=args.dis_count,
        stride=28,
        prefix="dis5k",
    )

    bg_all = duts_recs + dis_recs
    if not bg_all:
        raise RuntimeError("Could not download any benchmark images from Hugging Face datasets-server.")

    inpaint_recs = _build_inpainting_set(bg_all, count=min(args.inpaint_count, len(bg_all)))

    # Write standardized Txt2Img prompts
    prompts_path = DATASET_DIR / "txt2img_prompts.json"
    prompts_path.write_text(json.dumps(TXT2IMG_PROMPTS, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write dataset manifest for full reproducibility in KT 2 & KT 3
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bg_removal_count": len(bg_all),
        "bg_sources": {
            "nobg/DUTS (test)": len(duts_recs),
            "nobg/DIS5K (DIS_VD)": len(dis_recs),
        },
        "inpainting_count": len(inpaint_recs),
        "txt2img_prompts_count": len(TXT2IMG_PROMPTS),
        "bg_removal": bg_all,
        "inpainting": inpaint_recs,
    }
    manifest_path = DATASET_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    total_bytes = sum(f.stat().st_size for f in DATASET_DIR.rglob("*") if f.is_file())
    print("\n==========================================================================")
    print(f"[✓] Dataset ready at: {DATASET_DIR}")
    print(f"    Total size        : {total_bytes / (1024 * 1024):.1f} MB")
    print(f"    Background removal: {len(bg_all)} image+mask pairs")
    print(f"    Inpainting/Erasure: {len(inpaint_recs)} pairs")
    print(f"    Txt2Img prompts   : {len(TXT2IMG_PROMPTS)} prompts")
    print("==========================================================================")


if __name__ == "__main__":
    main()
