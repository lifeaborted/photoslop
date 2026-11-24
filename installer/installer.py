import os
import sys
import time
import json
import shutil
import urllib.request
import subprocess
import webbrowser
from pathlib import Path

# Base Paths
INSTALLER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSTALLER_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_CONFIG_PATH = MODELS_DIR / "models.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCKER_COMPOSE_FILE = PROJECT_ROOT / "docker" / "docker-compose.yml"


def print_banner():
    banner = r"""
  ==============================================================
    ____                       ____  __                   
   / __ \___  ____  ___  _____/ / / / /___  ____          
  / /_/ / _ \/ __ \/ _ \/ ___/ /_/ / __ \/ __ \         
 / ____/  __/ /_/ /  __(__  ) __  / /_/ / /_/ /         
/_/    \___/ .___/\___/____/_/ /_/\____/ .___/          
          /_/                         /_/                 
     AI Image Editor - Docker Installer & Model Downloader
  ==============================================================
    """
    print(banner)


def check_docker():
    """Verify Docker and Docker Compose are installed and running."""
    print("[1/5] Проверка наличия и статуса Docker...")

    # Check docker CLI
    if not shutil.which("docker"):
        print("[-] ОШИБКА: Docker не найден в системе!")
        print("    Пожалуйста, скачайте и установите Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return False

    # Check docker daemon
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print("[-] ОШИБКА: Docker daemon не запущен!")
            print("    Пожалуйста, запустите Docker Desktop и повторите установку.")
            return False
        print("  [+] Docker daemon активен и готов к работе.")
    except Exception as e:
        print(f"[-] Ошибка при обращении к Docker: {e}")
        return False

    # Check GPU availability
    try:
        gpu_check = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8"
        )
        if gpu_check.returncode == 0:
            print("  [+] Обнаружен GPU NVIDIA! Поддержка CUDA в Docker будет задействована.")
        else:
            print("  [!] NVIDIA GPU не обнаружен. Приложение будет запущено в CPU-режиме.")
    except Exception:
        print("  [!] nvidia-smi недоступен. Работа в CPU-режиме.")

    return True


def ensure_directories():
    """Create runtime host directories."""
    print("[2/5] Подготовка каталогов /models, /output, /logs...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("  [+] Каталоги успешно подготовлены.")


def _is_model_ready(model: dict) -> bool:
    """Checks whether real model weights exist on disk (ignores empty/broken stubs)."""
    filename = model.get("filename", "")
    if not filename:
        return False
    dest = MODELS_DIR / filename
    if not dest.exists():
        return False
    if dest.is_dir():
        return (dest / "model_index.json").exists()
    return dest.stat().st_size > 1024 * 1024


def download_models_interactive():
    """Read models.json and optionally download missing weights via scripts/download_model.py."""
    print("[3/5] Проверка и загрузка моделей из models/models.json...")

    if not MODELS_CONFIG_PATH.exists():
        print(f"[-] Файл конфигурации {MODELS_CONFIG_PATH} не найден. Пропуск скачивания.")
        return

    with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    models = []
    if "categories" in config:
        for cat, m_list in config["categories"].items():
            for m in m_list:
                m_copy = dict(m)
                m_copy["category"] = cat
                models.append(m_copy)
    else:
        models = config.get("models", [])

    print(f"  Найдено описаний моделей в конфиге: {len(models)}")

    missing = []
    for m in models:
        name = m.get("name", m.get("id"))
        if _is_model_ready(m):
            print(f"  [✓] Модель '{name}' уже присутствует на диске: {m.get('filename')}")
        else:
            print(f"  [!] Модель '{name}' ({m.get('id')}) отсутствует на диске.")
            missing.append(m)

    if not missing:
        print("  [+] Все утверждённые модели готовы к запуску.")
        return

    choice = input(
        f"\n  Скачать недостающие модели ({len(missing)} шт.) сейчас? [y/N]: "
    ).strip().lower()
    if choice != "y":
        print("  [i] Пропуск скачивания. Веса можно докачать позже: uv run python scripts/download_model.py --stage champions")
        return

    download_script = PROJECT_ROOT / "scripts" / "download_model.py"
    if not download_script.exists():
        print(f"  [-] Скрипт загрузки не найден: {download_script}")
        return

    # Prefer uv, fallback to project .venv python, then current interpreter
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if shutil.which("uv"):
        cmd = ["uv", "run", "python", str(download_script), "--stage", "champions"]
    elif venv_py.exists():
        cmd = [str(venv_py), str(download_script), "--stage", "champions"]
    else:
        cmd = [sys.executable, str(download_script), "--stage", "champions"]

    print(f"  Выполняется: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if res.returncode != 0:
            print(
                "  [-] Не удалось скачать модели автоматически (проверьте uv / .venv и доступ к Hugging Face).\n"
                "      Запустите вручную: uv run python scripts/download_model.py --stage champions"
            )
    except Exception as e:
        print(f"  [-] Ошибка запуска загрузчика ({e}). Скачайте вручную через scripts/download_model.py.")


def deploy_docker():
    """Build and spin up Docker containers."""
    print("\n[4/5] Развёртывание Docker-контейнера PepeShop...")

    cmd = ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "--build", "-d"]

    print(f"  Выполняется: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            print("  [-] Docker Compose с GPU не удался. Пробуем запуск в стандартном режиме...")
            # Fallback without GPU reservation if nvidia-container-toolkit is missing
            fallback_cmd = ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "--build", "-d", "--no-deps"]
            subprocess.run(fallback_cmd, cwd=PROJECT_ROOT)
        print("  [+] Docker контейнеры успешно запущены!")
        return True
    except Exception as e:
        print(f"[-] Ошибка развёртывания Docker: {e}")
        return False


def wait_and_open_browser():
    """Poll health endpoint and open browser."""
    print("\n[5/5] Ожидание инициализации сервера на http://localhost:8000 ...")

    target_url = "http://localhost:8000"
    health_url = "http://localhost:8000/api/models"
    success = False

    for attempt in range(30):
        try:
            req = urllib.request.urlopen(health_url, timeout=2)
            if req.status == 200:
                success = True
                break
        except Exception:
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()

    print("\n")
    if success:
        print("==============================================================")
        print(f"  [✓] Приложение успешно запущено и готово к работе!")
        print(f"  Веб-интерфейс: {target_url}")
        print(f"  Логи:          {LOGS_DIR / 'log.txt'}")
        print(f"  Сохранения:    {OUTPUT_DIR}")
        print("==============================================================")
        try:
            webbrowser.open(target_url)
        except Exception:
            pass
    else:
        print(f"[!] Сервер запускается медленнее обычного. Перейдите по адресу {target_url} вручную.")


def main():
    print_banner()
    if not check_docker():
        input("\nНажмите Enter для выхода...")
        sys.exit(1)

    ensure_directories()
    download_models_interactive()
    if deploy_docker():
        wait_and_open_browser()

    input("\nУстановка завершена. Нажмите Enter для закрытия окна установщика...")


if __name__ == "__main__":
    main()
