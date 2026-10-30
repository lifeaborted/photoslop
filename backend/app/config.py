import os
from pathlib import Path

# Base Paths (allowing override via environment variables for Docker)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))

MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
MODELS_CONFIG_PATH = MODELS_DIR / "models.json"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))
LOG_FILE_PATH = LOGS_DIR / "log.txt"

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", PROJECT_ROOT / "frontend"))

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Application Settings
APP_NAME = "PepeShop Local AI Editor"
APP_VERSION = "0.1.0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
