"""
Скрипт загрузки модели SDXL-Turbo INT8 с Hugging Face напрямую в папку models/
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def download_sdxl_turbo(target_dir: Path = MODELS_DIR / "sdxl-turbo-torch-int8"):
    print(f"=== Загрузка imgailab/sdxl-turbo-torch-int8 в {target_dir} ===")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] Библиотека huggingface_hub не установлена. Устанавливаем...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)

    print("Начало скачивания репозитория (суммарно ~4.18 ГБ)...")
    snapshot_download(
        repo_id="imgailab/sdxl-turbo-torch-int8",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # Важно для Windows
        ignore_patterns=["*.git*", "*.md"]  # Пропускаем ненужную документацию
    )
    print(f"\n[✓] Модель успешно загружена в: {target_dir}")
    print("Содержимое папки:")
    for item in target_dir.iterdir():
        print(f"  - {item.name}")

if __name__ == "__main__":
    download_sdxl_turbo()
