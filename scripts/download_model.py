#!/usr/bin/env python3
"""
PepeShop Stage-by-Stage Model Manager (<= 6 GB VRAM Edition)
Downloads only the models needed for the current testing stage and cleans up
temporary candidate weights after benchmarking to save SSD space.

Usage:
  python scripts/download_model.py --list
  python scripts/download_model.py --stage bg
  python scripts/download_model.py --stage txt2img
  python scripts/download_model.py --stage inpaint
  python scripts/download_model.py --model lama-cleaner-onnx
  python scripts/download_model.py --cleanup-candidates
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARK_CONFIG = MODELS_DIR / "models_benchmark.json"

STAGE_MAP = {
    "bg": ["background_removal"],
    "background_removal": ["background_removal"],
    "txt2img": ["txt2img"],
    "inpaint": ["object_removal", "inpaint"],
    "object_removal": ["object_removal"],
    "champions": None,  # special: only champion=True models
}

FP16_ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "feature_extractor/*",
    "text_encoder/*.json",
    "text_encoder/*.fp16.safetensors",
    "text_encoder_2/*.json",
    "text_encoder_2/*.fp16.safetensors",
    "unet/*.json",
    "unet/*.fp16.safetensors",
    "vae/*.json",
    "vae/*.fp16.safetensors",
]


def load_benchmark_models() -> List[Dict[str, Any]]:
    data = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    models: List[Dict[str, Any]] = []
    for cat, items in data.get("categories", {}).items():
        for m in items:
            entry = dict(m)
            entry["category"] = cat
            models.append(entry)
    return models


def is_model_downloaded(model: Dict[str, Any]) -> bool:
    filename = model.get("filename", "")
    if not filename:
        return False
    p = MODELS_DIR / filename
    if not p.exists():
        return False
    if p.is_dir():
        return (p / "model_index.json").exists()
    return p.stat().st_size > 1024 * 1024  # > 1 MB


def get_disk_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 1)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def print_status(models: List[Dict[str, Any]]) -> None:
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024 ** 3)
    print(f"\nDisk free on {PROJECT_ROOT.drive}: {free_gb:.2f} GB")
    print("=" * 98)
    print(f"{'Category':<20} {'Model ID':<26} {'Status':<12} {'Est.GB':<8} {'VRAM':<8} {'Role'}")
    print("-" * 98)
    for m in models:
        dl = is_model_downloaded(m)
        status = "DOWNLOADED" if dl else "missing"
        role = "CHAMPION" if m.get("champion") else "candidate"
        est_disk = f"{m.get('disk_gb', 0):.2f}"
        est_vram = "CPU" if m.get("est_vram_gb", 0) == 0 else f"~{m['est_vram_gb']:.1f}GB"
        print(f"{m['category']:<20} {m['id']:<26} {status:<12} {est_disk:<8} {est_vram:<8} {role}")
    print("=" * 98)


def download_single_model(model: Dict[str, Any]) -> None:
    model_id = model["id"]
    if is_model_downloaded(model):
        print(f"[✓] Already downloaded: {model_id} ({MODELS_DIR / model['filename']})")
        return

    print(f"\n>>> Downloading [{model['category']}] {model['name']} ({model_id}) ~{model.get('disk_gb', '?')} GB...")

    # 1. rembg ONNX models
    if model.get("rembg_model"):
        rembg_name = model["rembg_model"]
        rembg_home = MODELS_DIR / "rembg"
        rembg_home.mkdir(parents=True, exist_ok=True)
        os.environ["REMBG_HOME"] = str(rembg_home)
        from rembg.sessions import sessions
        if rembg_name not in sessions:
            raise ValueError(f"rembg session '{rembg_name}' not found in installed rembg package.")
        out_path = sessions[rembg_name].download_models()
        print(f"[✓] Downloaded rembg ONNX model '{rembg_name}' -> {out_path}")
        return

    # 2. Single ONNX file on Hugging Face (e.g. LaMa FP32)
    filename = model["filename"]
    repo_id = model.get("repo_id")
    if not repo_id:
        raise ValueError(f"Model '{model_id}' has neither rembg_model nor repo_id configured.")

    from huggingface_hub import hf_hub_download, snapshot_download

    if filename.endswith(".onnx"):
        out_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
        )
        print(f"[✓] Downloaded ONNX file -> {out_path} ({get_disk_size_mb(Path(out_path))} MB)")
        return

    # 3. Diffusers directory on Hugging Face
    target_dir = MODELS_DIR / filename
    target_dir.mkdir(parents=True, exist_ok=True)

    if model.get("variant") == "fp16":
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=FP16_ALLOW_PATTERNS,
        )
    else:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            ignore_patterns=["*.git*", "*.md"],
        )

    print(f"[✓] Downloaded Diffusers model -> {target_dir} ({get_disk_size_mb(target_dir)} MB)")


def delete_model(model: Dict[str, Any]) -> None:
    p = MODELS_DIR / model["filename"]
    if model.get("rembg_model"):
        p = MODELS_DIR / "rembg" / "models" / model["rembg_model"]
    if not p.exists():
        print(f"[-] Not present on disk: {model['id']}")
        return
    size_mb = get_disk_size_mb(p)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    print(f"[x] Deleted {model['id']} ({size_mb:.1f} MB freed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="PepeShop Stage-by-Stage Model Downloader")
    parser.add_argument("--list", action="store_true", help="Show download status of all benchmark models")
    parser.add_argument(
        "--stage",
        type=str,
        choices=list(STAGE_MAP.keys()),
        help="Download all models for a testing stage: bg, txt2img, inpaint, object_removal, champions",
    )
    parser.add_argument("--model", type=str, help="Download a single model by ID")
    parser.add_argument("--delete", type=str, help="Delete a single downloaded model by ID")
    parser.add_argument(
        "--cleanup-candidates",
        action="store_true",
        help="Delete all non-champion candidate models after benchmarking (keeps the 4 production champions)",
    )
    args = parser.parse_args()

    models = load_benchmark_models()

    if args.list or not any([args.stage, args.model, args.delete, args.cleanup_candidates]):
        print_status(models)
        return

    if args.delete:
        matches = [m for m in models if m["id"] == args.delete]
        if not matches:
            print(f"[!] Unknown model ID: {args.delete}")
            sys.exit(1)
        delete_model(matches[0])
        return

    if args.cleanup_candidates:
        print("Cleaning up temporary non-champion candidate models...")
        for m in models:
            if not m.get("champion"):
                delete_model(m)
        print_status(models)
        return

    targets: List[Dict[str, Any]] = []
    if args.model:
        targets = [m for m in models if m["id"] == args.model]
        if not targets:
            print(f"[!] Unknown model ID: {args.model}")
            sys.exit(1)
    elif args.stage == "champions":
        targets = [m for m in models if m.get("champion")]
    elif args.stage:
        cats = STAGE_MAP[args.stage]
        targets = [m for m in models if m["category"] in cats]

    total_est = sum(m.get("disk_gb", 0) for m in targets if not is_model_downloaded(m))
    print(f"Selected {len(targets)} model(s); estimated new download: ~{total_est:.2f} GB")

    for m in targets:
        download_single_model(m)

    print_status(models)


if __name__ == "__main__":
    main()
