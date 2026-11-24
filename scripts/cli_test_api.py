#!/usr/bin/env python3
"""
PepeShop Backend CLI Testing Utility
Allows fast verification of all backend endpoints from the terminal without using the web UI.

Examples:
  uv run python scripts/cli_test_api.py --status
  uv run python scripts/cli_test_api.py --models
  uv run python scripts/cli_test_api.py --switch --category background_removal --model birefnet-general-lite
  uv run python scripts/cli_test_api.py --task txt2img --prompt "a cyberpunk cat, 8k" --steps 1 --seed 42
  uv run python scripts/cli_test_api.py --task remove-bg --input output/test_sdxl_turbo_int8.png
  uv run python scripts/cli_test_api.py --task replace-bg --input input.png --bg-type color --bg-color "#1e293b"
  uv run python scripts/cli_test_api.py --task remove-object --input input.png --mask mask.png
  uv run python scripts/cli_test_api.py --task inpaint --input input.png --mask mask.png --prompt "wooden table"
  uv run python scripts/cli_test_api.py --task text-edit --input input.png --prompt "make it retro sepia"
"""

import argparse
import sys
import time
import base64
from pathlib import Path
import requests

API_URL = "http://localhost:8000"


def file_to_data_uri(path: Path) -> str:
    if not path.exists():
        print(f"[-] Error: file not found at {path}")
        sys.exit(1)
    suffix = path.suffix.lower().replace(".", "")
    if suffix == "jpg":
        suffix = "jpeg"
    mime = f"image/{suffix}" if suffix in ("png", "jpeg", "webp") else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def save_data_uri_to_file(data_uri: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "," in data_uri:
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    data = base64.b64decode(encoded)
    out_path.write_bytes(data)
    print(f"[+] Output saved to: {out_path} ({len(data) / 1024:.1f} KB)")


def check_status():
    try:
        r = requests.get(f"{API_URL}/api/models/hardware", timeout=5)
        if r.status_code == 200:
            hw = r.json()
            print("=== PepeShop Server & Hardware Telemetry ===")
            print(f"  GPU Available: {hw.get('gpu_available')}")
            print(f"  GPU Name     : {hw.get('gpu_name')}")
            print(f"  Driver       : {hw.get('driver_version')}")
            print(f"  VRAM         : {hw.get('vram_used_mb', 0):.0f}MB / {hw.get('vram_total_mb', 0):.0f}MB ({hw.get('vram_percent', 0)}%)")
        else:
            print(f"[-] Server returned status code {r.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[-] Cannot connect to backend at {API_URL}. Is 'python backend/run.py' running?")


def list_models():
    try:
        r = requests.get(f"{API_URL}/api/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"=== Registered Models ({data.get('total_count', 0)}) ===")
            for m in data.get("models", []):
                ready = "[READY]" if m.get("is_downloaded") else "[NOT DOWNLOADED]"
                size = f"({m.get('file_size_mb', 0):.1f} MB)" if m.get("is_downloaded") else ""
                print(f"  - [{m['category']}] {m['id']}: {m['name']} {ready} {size}")

            print("\n=== Active Models ===")
            for cat, info in data.get("active_models", {}).items():
                print(f"  - {cat:20} -> {info.get('id')} ({info.get('name')})")
    except Exception as e:
        print(f"[-] Failed to fetch models: {e}")


def switch_model(category: str, model_id: str):
    try:
        r = requests.post(f"{API_URL}/api/models/switch", json={"category": category, "model_id": model_id})
        if r.status_code == 200:
            print(f"[+] Model for [{category}] successfully switched to: {model_id}")
        else:
            print(f"[-] Switch failed ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"[-] Switch request error: {e}")


def main():
    parser = argparse.ArgumentParser(description="PepeShop Backend CLI Client")
    parser.add_argument("--status", action="store_true", help="Print server & hardware telemetry")
    parser.add_argument("--models", action="store_true", help="List all models and active status")
    parser.add_argument("--switch", action="store_true", help="Switch active model")
    parser.add_argument("--category", type=str, help="Category for model switch")
    parser.add_argument("--model", type=str, help="Model ID for model switch")

    parser.add_argument(
        "--task",
        choices=["txt2img", "remove-bg", "replace-bg", "remove-object", "inpaint", "text-edit"],
        help="Task to execute",
    )
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation or editing")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--mask", type=str, help="Path to mask image (for inpaint / remove-object)")
    parser.add_argument("--bg-type", choices=["color", "image", "prompt"], default="color", help="Background type for replace-bg")
    parser.add_argument("--bg-color", type=str, default="#ffffff", help="Hex color for replace-bg --bg-type color")
    parser.add_argument("--bg-image", type=str, help="Path to background image for replace-bg --bg-type image")
    parser.add_argument("--steps", type=int, default=1, help="Steps count")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Optional output path")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.models:
        list_models()
        return

    if args.switch:
        if not args.category or not args.model:
            print("[-] Usage: --switch --category <cat> --model <model_id>")
            return
        switch_model(args.category, args.model)
        return

    if not args.task:
        parser.print_help()
        return

    timestamp = int(time.time())

    # 1. Txt2Img
    if args.task == "txt2img":
        if not args.prompt:
            print("[-] --prompt is required for txt2img")
            return
        print(f"[*] Requesting txt2img: '{args.prompt}' (steps={args.steps}, seed={args.seed})...")
        r = requests.post(f"{API_URL}/api/generation/txt2img", json={
            "prompt": args.prompt,
            "steps": args.steps,
            "seed": args.seed,
            "width": 512,
            "height": 512
        })
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_txt2img_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Generation succeeded in {res['elapsed_seconds']}s (Model: {res['model_name']})")
        else:
            print(f"[-] Generation error {r.status_code}: {r.text}")

    # 2. Remove Background
    elif args.task == "remove-bg":
        if not args.input:
            print("[-] --input image is required for remove-bg")
            return
        data_uri = file_to_data_uri(Path(args.input))
        print(f"[*] Requesting background removal for: {args.input}...")
        r = requests.post(f"{API_URL}/api/generation/remove-background", json={"image_data": data_uri})
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_rembg_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Background removal finished in {res['elapsed_seconds']}s")
        else:
            print(f"[-] Remove-bg error {r.status_code}: {r.text}")

    # 2.5 Replace Background
    elif args.task == "replace-bg":
        if not args.input:
            print("[-] --input image is required for replace-bg")
            return
        payload = {
            "image_data": file_to_data_uri(Path(args.input)),
            "bg_type": args.bg_type,
            "bg_color": args.bg_color,
            "seed": args.seed,
        }
        if args.bg_type == "image":
            if not args.bg_image:
                print("[-] --bg-image is required when --bg-type image")
                return
            payload["bg_image_data"] = file_to_data_uri(Path(args.bg_image))
        elif args.bg_type == "prompt":
            if not args.prompt:
                print("[-] --prompt is required when --bg-type prompt")
                return
            payload["bg_prompt"] = args.prompt
        print(f"[*] Requesting replace-background ({args.bg_type}) for: {args.input}...")
        r = requests.post(f"{API_URL}/api/generation/replace-background", json=payload)
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_replace_bg_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Background replaced in {res['elapsed_seconds']}s")
        else:
            print(f"[-] Replace-bg error {r.status_code}: {r.text}")

    # 3. Clean Object Removal (LaMa)
    elif args.task == "remove-object":
        if not args.input or not args.mask:
            print("[-] Both --input and --mask are required for remove-object")
            return
        img_uri = file_to_data_uri(Path(args.input))
        mask_uri = file_to_data_uri(Path(args.mask))
        print(f"[*] Requesting object removal for: {args.input} with mask {args.mask}...")
        r = requests.post(f"{API_URL}/api/generation/remove-object", json={
            "image_data": img_uri,
            "mask_data": mask_uri
        })
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_object_removed_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Object erased in {res['elapsed_seconds']}s")
        else:
            print(f"[-] Remove-object error {r.status_code}: {r.text}")

    # 4. Inpainting
    elif args.task == "inpaint":
        if not args.input or not args.mask or not args.prompt:
            print("[-] --input, --mask, and --prompt are required for inpaint")
            return
        img_uri = file_to_data_uri(Path(args.input))
        mask_uri = file_to_data_uri(Path(args.mask))
        print(f"[*] Requesting inpainting: '{args.prompt}'...")
        r = requests.post(f"{API_URL}/api/generation/inpaint", json={
            "image_data": img_uri,
            "mask_data": mask_uri,
            "prompt": args.prompt,
            "steps": args.steps,
            "seed": args.seed
        })
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_inpaint_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Inpainting completed in {res['elapsed_seconds']}s")
        else:
            print(f"[-] Inpaint error {r.status_code}: {r.text}")

    # 5. Text-Guided Edit
    elif args.task == "text-edit":
        if not args.input or not args.prompt:
            print("[-] --input and --prompt are required for text-edit")
            return
        img_uri = file_to_data_uri(Path(args.input))
        print(f"[*] Requesting text-guided edit: '{args.prompt}'...")
        r = requests.post(f"{API_URL}/api/generation/text-edit", json={
            "image_data": img_uri,
            "prompt": args.prompt,
            "seed": args.seed
        })
        if r.status_code == 200:
            res = r.json()
            out_file = Path(args.out) if args.out else Path(f"output/cli_edit_{timestamp}.png")
            save_data_uri_to_file(res["data_uri"], out_file)
            print(f"[+] Text-guided edit completed in {res['elapsed_seconds']}s")
        else:
            print(f"[-] Text-edit error {r.status_code}: {r.text}")


if __name__ == "__main__":
    main()
