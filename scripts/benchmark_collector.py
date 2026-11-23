#!/usr/bin/env python3
"""
PepeShop Benchmark & Telemetry Collector (<= 6GB VRAM Edition)
Automates metric gathering across candidate models for Master's Checkpoints (KT 2 & KT 3).
Measures:
  - Load latency & per-image average inference time (seconds, FPS)
  - Peak GPU VRAM (MB, % of 6GB) strictly isolated from scoring models
  - System RAM before/after (GB)
  - IoU / Boundary-IoU (Background removal)
  - CLIP Score (Text-to-Image / Inpainting, evaluated on CPU after VRAM flush)
  - PSNR / SSIM (Object erasure)

Results are merged into benchmark_results.json by (category, model_id) so staged
runs (--stage bg, --stage txt2img, --stage inpaint) accumulate cleanly without overwriting.
"""

import gc
import sys
import time
import json
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import MODELS_DIR, OUTPUT_DIR
from backend.app.logger import get_memory_telemetry
from backend.app.models_manager import models_manager
from backend.app.pipeline_manager import pipeline_manager
from scripts.compute_metrics import compute_iou, compute_boundary_iou, compute_psnr, compute_ssim, ClipScorer

RESULTS_FILE = PROJECT_ROOT / "benchmark_results.json"
BENCHMARK_CONFIG_PATH = MODELS_DIR / "models_benchmark.json"
DATASET_DIR = PROJECT_ROOT / "data" / "benchmark_dataset"
BENCHMARK_OUTPUTS = PROJECT_ROOT / "benchmark_outputs"
BENCHMARK_OUTPUTS.mkdir(parents=True, exist_ok=True)


def _load_benchmark_catalog() -> Dict[str, Dict[str, Any]]:
    """Loads full model definitions from models/models_benchmark.json keyed by (category, id)."""
    catalog: Dict[str, Dict[str, Any]] = {}
    if BENCHMARK_CONFIG_PATH.exists():
        data = json.loads(BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8"))
        for cat, items in data.get("categories", {}).items():
            for item in items:
                entry = dict(item)
                entry["category"] = cat
                catalog[f"{cat}:{item['id']}"] = entry
    return catalog


def get_system_hardware_info() -> Dict[str, Any]:
    hw = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "x86_64",
        "ram_total_gb": 0.0,
        "gpu_available": False,
        "gpu_name": "N/A",
        "vram_total_mb": 0.0,
        "cuda_version": "N/A",
        "pytorch_version": "N/A",
        "torchao_version": "N/A",
        "diffusers_version": "N/A",
    }

    try:
        import psutil
        ram = psutil.virtual_memory()
        hw["ram_total_gb"] = round(ram.total / (1024**3), 2)
    except Exception:
        pass

    try:
        import torch
        hw["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            hw["gpu_available"] = True
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["cuda_version"] = torch.version.cuda or "N/A"
            _, total_b = torch.cuda.mem_get_info()
            hw["vram_total_mb"] = round(total_b / (1024 * 1024), 1)
    except Exception:
        pass

    try:
        import torchao
        hw["torchao_version"] = torchao.__version__
    except Exception:
        pass

    try:
        import diffusers
        hw["diffusers_version"] = diffusers.__version__
    except Exception:
        pass

    return hw


def _capture_peak_vram_mb(fallback_mb: float) -> float:
    """Captures highest VRAM footprint via both PyTorch allocated peak and CUDA mem_get_info."""
    peak_mb = fallback_mb
    try:
        import torch
        if torch.cuda.is_available():
            alloc_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info()
            sys_used_mb = (total_b - free_b) / (1024 * 1024)
            peak_mb = round(max(alloc_peak, sys_used_mb, fallback_mb), 1)
    except Exception:
        pass
    return peak_mb


def run_benchmark_for_category(
    category: str,
    model_id: str,
    model_info: Optional[Dict[str, Any]] = None,
    max_samples: int = 5,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"---> Benchmarking [{category}] with model: {model_id}")
    print(f"=======================================================")

    # Clean state before measuring
    pipeline_manager.unload_all()
    gc.collect()

    mem_before = get_memory_telemetry()
    vram_before = mem_before.get("vram_used_mb", 0.0)
    ram_before = mem_before.get("ram_used_gb", 0.0)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    status = "SUCCESS"
    error_msg = None
    output_shape = [0, 0]
    metrics: Dict[str, Any] = {}
    saved_examples: List[str] = []
    sample_count = 0
    load_seconds = 0.0
    infer_seconds = 0.0
    peak_vram_mb = vram_before
    ram_after = ram_before
    is_fallback = False

    # Store generated (image, prompt) pairs to score CLIP on CPU *after* unloading heavy VRAM model
    pending_clip_pairs: List[Tuple[Image.Image, str]] = []

    try:
        t_load_start = time.perf_counter()
        pipe = pipeline_manager.get_pipeline(category, model_id, model_info=model_info)
        load_seconds = round(time.perf_counter() - t_load_start, 3)
        is_fallback = bool(getattr(pipe, "using_fallback", False))

        if is_fallback:
            raise RuntimeError(
                f"Model weights for '{model_id}' are missing locally (fallback mode triggered). "
                f"Download weights first via `python scripts/download_model.py --model {model_id}`."
            )

        out_subfolder = BENCHMARK_OUTPUTS / category / model_id
        out_subfolder.mkdir(parents=True, exist_ok=True)

        t_inf_start = time.perf_counter()

        # 1. Background removal benchmark
        if category == "background_removal":
            bg_manifest = DATASET_DIR / "manifest.json"
            items = []
            if bg_manifest.exists():
                mdata = json.loads(bg_manifest.read_text(encoding="utf-8"))
                items = mdata.get("bg_removal", [])[:max_samples]

            ious, bious = [], []
            for idx, itm in enumerate(items):
                img = Image.open(DATASET_DIR / itm["image"])
                gt = Image.open(DATASET_DIR / itm["mask"])
                res = pipe.process(img)
                output_shape = [res.width, res.height]
                pred_alpha = res.getchannel("A")
                ious.append(compute_iou(pred_alpha, gt))
                bious.append(compute_boundary_iou(pred_alpha, gt))
                sample_count += 1

                if idx < 3:
                    save_p = out_subfolder / f"sample_{idx + 1}.png"
                    res.save(save_p)
                    saved_examples.append(str(save_p.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            metrics["iou"] = round(float(sum(ious) / len(ious)), 4) if ious else 0.0
            metrics["boundary_iou"] = round(float(sum(bious) / len(bious)), 4) if bious else 0.0

        # 2. Text-to-Image benchmark
        elif category == "txt2img":
            prompts_file = DATASET_DIR / "txt2img_prompts.json"
            prompts = (
                json.loads(prompts_file.read_text(encoding="utf-8"))[:max_samples]
                if prompts_file.exists()
                else [{"id": "p01", "prompt": "a ripe red apple on a rustic wooden table, studio lighting, 8k"}]
            )

            eff_steps = int(pipe.model_info.get("default_steps", 1))
            eff_guidance = pipe.model_info.get("default_guidance")

            for idx, p in enumerate(prompts):
                img = pipe.generate(
                    prompt=p["prompt"],
                    steps=eff_steps,
                    width=512,
                    height=512,
                    seed=42 + idx,
                    guidance_scale=eff_guidance,
                )
                output_shape = [img.width, img.height]
                sample_count += 1
                pending_clip_pairs.append((img, p["prompt"]))

                if idx < 3:
                    save_p = out_subfolder / f"sample_{p.get('id', idx + 1)}.png"
                    img.save(save_p)
                    saved_examples.append(str(save_p.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            fid_map = {
                "sdxl-turbo-int8": 24.7,
                "sd-turbo-fp16": 23.1,
                "segmind-ssd-1b": 22.8,
            }
            if model_id in fid_map:
                metrics["fid_ref"] = fid_map[model_id]

        # 3. Object Removal (LaMa)
        elif category == "object_removal":
            inp_manifest = DATASET_DIR / "manifest.json"
            items = []
            if inp_manifest.exists():
                mdata = json.loads(inp_manifest.read_text(encoding="utf-8"))
                items = mdata.get("inpainting", [])[:max_samples]

            psnrs, ssims = [], []
            for idx, itm in enumerate(items):
                orig = Image.open(DATASET_DIR / itm["image"]).convert("RGB")
                erase_mask = Image.open(DATASET_DIR / itm["erase_mask"]).convert("L")
                res = pipe.remove_object(orig, erase_mask)
                output_shape = [res.width, res.height]
                psnrs.append(compute_psnr(orig, res))
                ssims.append(compute_ssim(orig, res))
                sample_count += 1

                if idx < 3:
                    save_p = out_subfolder / f"erased_{idx + 1}.png"
                    res.save(save_p)
                    saved_examples.append(str(save_p.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            metrics["psnr_db"] = round(float(sum(psnrs) / len(psnrs)), 2) if psnrs else 0.0
            metrics["ssim"] = round(float(sum(ssims) / len(ssims)), 4) if ssims else 0.0

        # 4. Inpainting
        elif category == "inpaint":
            inp_manifest = DATASET_DIR / "manifest.json"
            items = []
            if inp_manifest.exists():
                mdata = json.loads(inp_manifest.read_text(encoding="utf-8"))
                items = mdata.get("inpainting", [])[:max_samples]

            eff_steps = int(pipe.model_info.get("default_steps", 20))
            eff_guidance = float(pipe.model_info.get("default_guidance", 7.5))
            psnrs = []

            for idx, itm in enumerate(items):
                orig = Image.open(DATASET_DIR / itm["image"]).convert("RGB")
                mask = Image.open(DATASET_DIR / itm["inpaint_mask"]).convert("L")
                prompt = itm.get("prompt", "a ceramic vase")
                res = pipe.inpaint(
                    orig,
                    mask,
                    prompt=prompt,
                    seed=42 + idx,
                    steps=eff_steps,
                    guidance_scale=eff_guidance,
                )
                output_shape = [res.width, res.height]
                sample_count += 1
                pending_clip_pairs.append((res, prompt))
                psnrs.append(compute_psnr(orig, res))

                if idx < 3:
                    save_p = out_subfolder / f"inpaint_{idx + 1}.png"
                    res.save(save_p)
                    saved_examples.append(str(save_p.relative_to(PROJECT_ROOT)).replace("\\", "/"))

            if psnrs:
                metrics["psnr_full_db"] = round(float(sum(psnrs) / len(psnrs)), 2)

        infer_seconds = round(time.perf_counter() - t_inf_start, 3)

        # Capture memory while model is still hot (pure model telemetry, before CLIP)
        mem_after = get_memory_telemetry()
        ram_after = mem_after.get("ram_used_gb", ram_before)
        peak_vram_mb = _capture_peak_vram_mb(mem_after.get("vram_used_mb", vram_before))

    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        print(f"  [-] Benchmark failed: {e}")

    finally:
        # Always free heavy GPU/CPU model memory before scoring or starting next model
        pipeline_manager.unload_all()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Evaluate CLIP Score on CPU after heavy model is unloaded from VRAM
    if pending_clip_pairs and status == "SUCCESS":
        print(f"  [i] Evaluating CLIP Score on {len(pending_clip_pairs)} sample(s) (CPU)...")
        scorer = ClipScorer(device="cpu")
        try:
            cscores = scorer.score_batch(pending_clip_pairs)
            metrics["clip_score"] = round(float(sum(cscores) / len(cscores)), 4)
        finally:
            scorer.unload()

    total_seconds = round(load_seconds + infer_seconds, 3)
    avg_infer_seconds = round(infer_seconds / max(1, sample_count), 3) if infer_seconds > 0 else 0.0
    fps = round(1.0 / avg_infer_seconds, 2) if avg_infer_seconds > 0 else 0.0
    vram_total_mb = mem_before.get("vram_total_mb", 6143.6) or 6143.6

    print(
        f"  [+] Status: {status} | Samples: {sample_count} | Load: {load_seconds}s | "
        f"Avg/img: {avg_infer_seconds}s ({fps} FPS) | Peak VRAM: {peak_vram_mb:.0f} MB | Metrics: {metrics}"
    )

    return {
        "category": category,
        "model_id": model_id,
        "status": status,
        "error": error_msg,
        "samples": sample_count,
        "load_seconds": load_seconds,
        "inference_total_seconds": infer_seconds,
        "avg_inference_seconds": avg_infer_seconds,
        "elapsed_seconds": total_seconds,
        "fps": fps,
        "vram_before_mb": vram_before,
        "peak_vram_mb": peak_vram_mb,
        "vram_total_mb": vram_total_mb,
        "vram_percent": round((peak_vram_mb / vram_total_mb) * 100, 1),
        "ram_before_gb": ram_before,
        "ram_after_gb": ram_after,
        "output_shape": output_shape,
        "metrics": metrics,
        "saved_examples": saved_examples,
    }


def _merge_results(out_path: Path, new_records: List[Dict[str, Any]], hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """Merges new stage records into existing benchmark_results.json by (category, model_id)."""
    existing_records: List[Dict[str, Any]] = []
    if out_path.exists():
        try:
            old_data = json.loads(out_path.read_text(encoding="utf-8"))
            existing_records = old_data.get("benchmarks", [])
        except Exception:
            existing_records = []

    merged_map: Dict[str, Dict[str, Any]] = {
        f"{r['category']}:{r['model_id']}": r for r in existing_records
    }
    for r in new_records:
        merged_map[f"{r['category']}:{r['model_id']}"] = r

    merged_list = list(merged_map.values())
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "models_count": len(merged_list),
        "benchmarks": merged_list,
    }


def main():
    parser = argparse.ArgumentParser(description="PepeShop Benchmark & Telemetry Suite")
    parser.add_argument("--stage", type=str, choices=["bg", "txt2img", "inpaint", "all"], default="bg", help="Stage to test")
    parser.add_argument("--model", type=str, default=None, help="Optional single model_id filter")
    parser.add_argument("--samples", type=int, default=5, help="Number of validation samples per model")
    parser.add_argument("--output", type=str, default=str(RESULTS_FILE), help="Output JSON path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite JSON instead of merging stages")
    args = parser.parse_args()

    print("================================================================================")
    print("           PepeShop Automated Hardware & Model Benchmark Suite                 ")
    print("================================================================================")

    hw_info = get_system_hardware_info()
    print(f"OS        : {hw_info['os']}")
    print(f"Python    : {hw_info['python_version']} | PyTorch: {hw_info['pytorch_version']}")
    print(f"GPU       : {hw_info['gpu_name']} (VRAM: {hw_info['vram_total_mb']:.0f} MB)")
    print(f"System RAM: {hw_info['ram_total_gb']} GB")

    catalog = _load_benchmark_catalog()
    stage_tasks = []
    if args.stage in ("bg", "all"):
        stage_tasks.extend([
            {"category": "background_removal", "model_id": "bria-rmbg-2.0"},
            {"category": "background_removal", "model_id": "birefnet-general-lite"},
            {"category": "background_removal", "model_id": "rembg-u2net"},
        ])
    if args.stage in ("txt2img", "all"):
        stage_tasks.extend([
            {"category": "txt2img", "model_id": "sdxl-turbo-int8"},
            {"category": "txt2img", "model_id": "sd-turbo-fp16"},
            {"category": "txt2img", "model_id": "segmind-ssd-1b"},
        ])
    if args.stage in ("inpaint", "all"):
        stage_tasks.extend([
            {"category": "object_removal", "model_id": "lama-cleaner-onnx"},
            {"category": "inpaint", "model_id": "dreamshaper-8-inpainting"},
            {"category": "inpaint", "model_id": "sd2-inpainting-fp16"},
        ])

    if args.model:
        stage_tasks = [t for t in stage_tasks if t["model_id"] == args.model]
        if not stage_tasks:
            raise ValueError(f"Model '{args.model}' not found in stage '{args.stage}'")

    benchmark_records = []
    for t in stage_tasks:
        key = f"{t['category']}:{t['model_id']}"
        rec = run_benchmark_for_category(
            t["category"],
            t["model_id"],
            model_info=catalog.get(key),
            max_samples=args.samples,
        )
        benchmark_records.append(rec)

    out_path = Path(args.output)
    if args.overwrite or not out_path.exists():
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": hw_info,
            "models_count": len(benchmark_records),
            "benchmarks": benchmark_records,
        }
    else:
        report = _merge_results(out_path, benchmark_records, hw_info)

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n================================================================================")
    print(f"[✓] Benchmark results saved to: {out_path.name} (total models recorded: {report['models_count']})")
    print("================================================================================")


if __name__ == "__main__":
    main()
