import subprocess
import sys
import shutil
from pathlib import Path

INSTALLER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INSTALLER_DIR.parent
SCRIPT_PATH = INSTALLER_DIR / "installer.py"
OUTPUT_EXE_DIR = INSTALLER_DIR / "dist"

def build():
    print("=== Сборка бинарного установщика PepeShop Installer ===")

    # Check if pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller не найден. Установка pyinstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed" if False else "--console",
        "--name", "pepeshop-installer",
        "--distpath", str(OUTPUT_EXE_DIR),
        "--workpath", str(INSTALLER_DIR / "build"),
        "--specpath", str(INSTALLER_DIR),
        str(SCRIPT_PATH)
    ]

    print(f"Выполняется команда: {' '.join(cmd)}")
    res = subprocess.run(cmd)

    if res.returncode == 0:
        print("\n[✓] Бинарный установщик успешно собран!")
        print(f"Путь к исполняемому файлу: {OUTPUT_EXE_DIR / 'pepeshop-installer'}")
    else:
        print("\n[-] Ошибка сборки установщика.")

if __name__ == "__main__":
    build()
