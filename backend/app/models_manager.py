import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from backend.app.config import MODELS_DIR, MODELS_CONFIG_PATH
from backend.app.logger import logger


class ModelsManager:
    def __init__(self, config_path: Path = MODELS_CONFIG_PATH, models_dir: Path = MODELS_DIR):
        self.config_path = config_path
        self.models_dir = models_dir
        self.active_models: Dict[str, str] = {
            "inpaint": "sd15-inpainting-fp16",
            "edit": "flux1-kontext-q2",
            "txt2img": "sdxl-turbo-int8",
            "background_removal": "bria-rmbg-1.4"
        }
        self.cached_config: Dict[str, Any] = {}
        self.flattened_models: List[Dict[str, Any]] = []
        self._last_mtime: float = 0.0
        self.reload_config(force=True)

    def reload_config(self, force: bool = False) -> Dict[str, Any]:
        """Reads and updates configuration from models/models.json supporting categorized format."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Initializing default.")
            self._create_default_config()

        try:
            current_mtime = self.config_path.stat().st_mtime
            if not force and self.cached_config and current_mtime == self._last_mtime:
                return self.cached_config

            self._last_mtime = current_mtime
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            flattened = []
            categories = data.get("categories", {})

            # Support new categorized format
            if categories:
                for cat, models in categories.items():
                    for model in models:
                        m_copy = dict(model)
                        m_copy["category"] = cat

                        filename = m_copy.get("filename")
                        if filename:
                            path_obj = self.models_dir / filename
                            is_present = path_obj.exists()
                            m_copy["is_downloaded"] = is_present

                            if is_present:
                                if path_obj.is_file():
                                    m_copy["file_size_mb"] = round(path_obj.stat().st_size / (1024 * 1024), 2)
                                elif path_obj.is_dir():
                                    # Calculate directory size (e.g. for Diffusers repos like sdxl-turbo-torch-int8)
                                    total_bytes = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                                    m_copy["file_size_mb"] = round(total_bytes / (1024 * 1024), 2)
                            else:
                                m_copy["file_size_mb"] = 0
                        else:
                            m_copy["is_downloaded"] = False

                        flattened.append(m_copy)

                    # Auto-assign active model if not set
                    if cat not in self.active_models and models:
                        self.active_models[cat] = models[0]["id"]

            elif "models" in data:
                # Fallback for flat models list
                for model in data["models"]:
                    m_copy = dict(model)
                    filename = m_copy.get("filename")
                    if filename:
                        path_obj = self.models_dir / filename
                        m_copy["is_downloaded"] = path_obj.exists()
                        if path_obj.exists():
                            m_copy["file_size_mb"] = round(path_obj.stat().st_size / (1024 * 1024), 2)
                    flattened.append(m_copy)

            self.cached_config = data
            self.flattened_models = flattened
            logger.info(f"Loaded {len(flattened)} models across {len(categories)} categories from {self.config_path}")
            return self.cached_config
            return self.cached_config
        except Exception as e:
            logger.error(f"Error loading models configuration: {e}")
            return self.cached_config

    def get_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        self.reload_config()
        if category:
            return [m for m in self.flattened_models if m.get("category") == category]
        return self.flattened_models

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        for m in self.get_models():
            if m.get("id") == model_id:
                return m
        return None

    def get_active_models(self) -> Dict[str, Any]:
        result = {}
        for cat, model_id in self.active_models.items():
            model_info = self.get_model_by_id(model_id)
            result[cat] = {
                "id": model_id,
                "name": model_info["name"] if model_info else model_id,
                "details": model_info
            }
        return result

    def switch_model(self, category: str, model_id: str) -> Dict[str, Any]:
        """Dynamically switch active model for a specific task category."""
        model = self.get_model_by_id(model_id)
        if not model:
            msg = f"Model '{model_id}' not found in configuration!"
            logger.error(msg)
            raise ValueError(msg)

        old_model_id = self.active_models.get(category)
        self.active_models[category] = model_id

        logger.info(
            f"Dynamic model switch: Category [{category}] changed from '{old_model_id}' to '{model_id}' ({model.get('name')})"
        )

        return {
            "category": category,
            "previous_model": old_model_id,
            "active_model": model_id,
            "model_info": model,
            "status": "success"
        }

    def get_hardware_status(self) -> Dict[str, Any]:
        """Reads real-time VRAM, RAM, and GPU status."""
        status = {
            "gpu_available": False,
            "gpu_name": "N/A",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
            "vram_percent": 0.0,
            "driver_version": "N/A",
            "backend_device": "cpu"
        }

        # Try nvidia-smi
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                timeout=2
            ).strip()
            if output:
                lines = output.splitlines()
                first_gpu = lines[0].split(",")
                if len(first_gpu) >= 5:
                    status["gpu_available"] = True
                    status["gpu_name"] = first_gpu[0].strip()
                    total = float(first_gpu[1].strip())
                    used = float(first_gpu[2].strip())
                    free = float(first_gpu[3].strip())
                    status["vram_total_mb"] = total
                    status["vram_used_mb"] = used
                    status["vram_free_mb"] = free
                    status["vram_percent"] = round((used / total) * 100, 1) if total > 0 else 0
                    status["driver_version"] = first_gpu[4].strip()
                    status["backend_device"] = "cuda"
        except Exception:
            pass

        return status

    def _create_default_config(self):
        """Creates minimal categorized configuration if models.json is missing."""
        default_data = {
            "categories": {
                "inpaint": [],
                "edit": [],
                "txt2img": [],
                "background_removal": []
            }
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=2, ensure_ascii=False)


# Global singleton instance
models_manager = ModelsManager()
