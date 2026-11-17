#!/usr/bin/env python3
"""
PepeShop Benchmark & Telemetry Collector
Automates metric gathering across candidate models for Master's Checkpoints (KT 2 & KT 3).
Saves telemetry into benchmark_results.json for automatic report generation.

Usage:
  python scripts/benchmark_collector.py
  python scripts/benchmark_collector.py --quick
  python scripts/benchmark_collector.py --category txt2img
"""

import sys
import time
import json
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import MODELS_DIR, OUTPUT_DIR
from backend.app.logger import get_memory_telemetry
from backend.app.models_manager import models_manager
from backend.app.pipeline_manager import pipeline_manager


RESULTS_FILE = PROJECT_ROOT / "benchmark_results.json"


def get_system_hardware_info() -> Dict[str, Any]:
    """Collects detailed hardware and software environment specification."""
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
        "diffusers_version": "N/A"
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
            free_b, total_b = torch.cuda.mem_get_info()
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


def run_benchmark_for_category(category: str, model_id: str, payload_options: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a single model benchmark and captures latency, peak VRAM, RAM, and image stats."""
    print(f"\n---> Benchmarking [{category}] with model: {model_id}...")

    # Pre-inference telemetry
    mem_before = get_memory_telemetry()
    vram_before = mem_before.get("vram_used_mb", 0.0)
    ram_before = mem_before.get("ram_used_gb", 0.0)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass

    start_time = time.perf_counter()
    status = "SUCCESS"
    error_msg = None
    output_shape = [0, 0]
    result_img = None

    try:
        pipe = pipeline_manager.get_pipeline(category, model_id)

        if category == "txt2img":
            result_img = pipe.generate(
                prompt=payload_options.get("prompt", "a red sports car, studio lighting, 8k"),
                steps=payload_options.get("steps", 1),
                width=payload_options.get("width", 512),
                height=payload_options.get("height", 512),
                seed=payload_options.get("seed", 42)
            )
        elif category == "background_removal":
            test_img = payload_options.get("image") or Image.new("RGBA", (512, 512), (100, 150, 200, 255))
            result_img = pipe.process(test_img)
        elif category == "object_removal":
            test_img = payload_options.get("image") or Image.new("RGB", (512, 512), (120, 120, 120))
            test_mask = payload_options.get("mask") or Image.new("L", (512, 512), 0)
            result_img = pipe.remove_object(test_img, test_mask)
        elif category in ("inpaint", "edit"):
            test_img = payload_options.get("image") or Image.new("RGB", (512, 512), (120, 120, 120))
            test_mask = payload_options.get("mask") or Image.new("L", (512, 512), 0)
            result_img = pipe.inpaint(
                image=test_img,
                mask=test_mask,
                prompt=payload_options.get("prompt", "golden statue"),
                steps=payload_options.get("steps", 20)
            )

        output_shape = [result_img.width, result_img.height] if result_img else [0, 0]
    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        print(f"  [-] Benchmark failed: {e}")

    elapsed = round(time.perf_counter() - start_time, 3)

    # Post-inference telemetry
    mem_after = get_memory_telemetry()
    vram_after = mem_after.get("vram_used_mb", 0.0)
    ram_after = mem_after.get("ram_used_gb", 0.0)

    peak_vram_mb = vram_after
    try:
        import torch
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            if peak_bytes > 0:
                peak_vram_mb = round(peak_bytes / (1024 * 1024), 1)
    except Exception:
        pass

    fps = round(1.0 / elapsed, 2) if elapsed > 0 else 0.0

    print(f"  [+] Status: {status} | Time: {elapsed}s ({fps} FPS) | Peak VRAM: {peak_vram_mb:.0f} MB")

    return {
        "category": category,
        "model_id": model_id,
        "status": status,
        "error": error_msg,
        "elapsed_seconds": elapsed,
        "fps": fps,
        "vram_before_mb": vram_before,
        "peak_vram_mb": peak_vram_mb,
        "vram_total_mb": mem_after.get("vram_total_mb", 6144.0),
        "vram_percent": round((peak_vram_mb / mem_after.get("vram_total_mb", 6144.0)) * 100, 1) if mem_after.get("vram_total_mb") else 0,
        "ram_before_gb": ram_before,
        "ram_after_gb": ram_after,
        "output_shape": output_shape
    }


def main():
    parser = argparse.ArgumentParser(description="PepeShop Benchmark & Telemetry Suite")
    parser.add_argument("--quick", action="store_true", help="Quick run with active production models")
    parser.add_argument("--category", type=str, help="Specific category to benchmark")
    parser.add_argument("--output", type=str, default=str(RESULTS_FILE), help="Output JSON path")
    args = parser.parse_args()

    print("================================================================================")
    print("           PepeShop Automated Hardware & Model Benchmark Suite                 ")
    print("================================================================================")

    hw_info = get_system_hardware_info()
    print(f"OS        : {hw_info['os']}")
    print(f"Python    : {hw_info['python_version']} | PyTorch: {hw_info['pytorch_version']}")
    print(f"GPU       : {hw_info['gpu_name']} (VRAM: {hw_info['vram_total_mb']:.0f} MB)")
    print(f"System RAM: {hw_info['ram_total_gb']} GB")

    benchmark_records = []

    # Benchmark tasks
    tasks = [
        {"category": "txt2img", "model_id": "sdxl-turbo-int8", "options": {"steps": 1, "prompt": "a red sports car on wet asphalt, 8k"}},
        {"category": "background_removal", "model_id": "bria-rmbg-2.0", "options": {}},
        {"category": "object_removal", "model_id": "lama-cleaner-onnx", "options": {}},
        {"category": "inpaint", "model_id": "flux1-fill-dev-q2", "options": {"steps": 20, "prompt": "cyberpunk neon lights"}}
    ]

    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]

    for t in tasks:
        rec = run_benchmark_for_category(t["category"], t["model_id"], t["options"])
        benchmark_records.append(rec)

    # Save results to JSON
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "models_count": len(benchmark_records),
        "benchmarks": benchmark_records
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n================================================================================")
    print(f"[✓] Benchmark results saved to: {out_path.name}")
    print("    You can now ask the AI to update KT2/KT3 reports based on these exact numbers!")
    print("================================================================================")


if __name__ == "__main__":
    main()
