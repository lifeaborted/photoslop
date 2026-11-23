import gc
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageFilter
import numpy as np
from backend.app.config import MODELS_DIR
from backend.app.logger import logger
from backend.app.pipelines.base import BasePipeline

REMBG_ID_MAP = {
    "bria-rmbg-2.0": "bria-rmbg",
    "birefnet-general-lite": "birefnet-general-lite",
    "birefnet-general": "birefnet-general",
    "isnet-general-use": "isnet-general-use",
    "rembg-u2net": "u2net",
}


class BackgroundRemovalPipeline(BasePipeline):
    """
    Pipeline for AI background removal using rembg 2.x ONNX sessions:
    - BRIA RMBG-2.0 (bria-rmbg ONNX, BiRefNet SOTA)
    - BiRefNet General Lite (ONNX)
    - U2-Net / IS-Net (CPU lightweight baseline)
    Runs reliably on CPU (satisfying the project spec IoU >= 0.9 without GPU) and GPU if CUDA ORT is present.
    """

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.rembg_session = None
        self.rembg_name: Optional[str] = None

    def _resolve_rembg_name(self) -> Optional[str]:
        if self.model_info.get("rembg_model"):
            return self.model_info["rembg_model"]
        model_id = self.model_info.get("id", "")
        return REMBG_ID_MAP.get(model_id)

    def _is_rembg_downloaded(self, rembg_name: str) -> bool:
        rembg_home = MODELS_DIR / "rembg"
        os.environ["REMBG_HOME"] = str(rembg_home)
        filename = self.model_info.get("filename")
        if filename and (MODELS_DIR / filename).exists():
            return True
        expected = rembg_home / "models" / rembg_name / f"{rembg_name}.onnx"
        if expected.exists():
            return True
        legacy = Path.home() / ".u2net" / f"{rembg_name}.onnx"
        if legacy.exists():
            return True
        default_rembg = Path.home() / ".rembg" / "models" / rembg_name / f"{rembg_name}.onnx"
        return default_rembg.exists()

    def load(self) -> bool:
        model_id = self.model_info.get("id", "")
        model_name = self.model_info.get("name", "")
        logger.info(f"Loading background removal model: {model_name} [{model_id}]")

        rembg_name = self._resolve_rembg_name()
        if rembg_name:
            os.environ["REMBG_HOME"] = str(MODELS_DIR / "rembg")
            if self._is_rembg_downloaded(rembg_name):
                try:
                    import onnxruntime as ort
                    from rembg import new_session

                    avail = ort.get_available_providers()
                    force_cpu = bool(self.model_info.get("force_cpu", False))
                    if not force_cpu and "CUDAExecutionProvider" in avail:
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]

                    logger.info(f"Creating rembg session '{rembg_name}' with providers={providers}")
                    self.rembg_session = new_session(rembg_name, providers=providers)
                    self.rembg_name = rembg_name
                    self.using_fallback = False
                    self.is_loaded = True
                    logger.info(f"Background removal model [{model_id}] loaded successfully ({rembg_name})")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load rembg session '{rembg_name}': {e}")
                    raise
            else:
                logger.warning(
                    f"ONNX weights for '{rembg_name}' not found locally. "
                    f"Run `python scripts/download_model.py --model {model_id}` to download."
                )

        # Fallback heuristic mode (flagged explicitly so benchmarks never accept it)
        self.using_fallback = True
        self.is_loaded = True
        logger.warning(f"Background removal [{model_id}] operating in FALLBACK algorithmic mode.")
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading background removal model: {self.model_info.get('name')}")
        if self.rembg_session is not None:
            del self.rembg_session
            self.rembg_session = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.is_loaded = False
        self.using_fallback = False
        return True

    def process(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """Removes background from PIL Image and returns RGBA Image."""
        start_time = time.time()
        model_id = self.model_info.get("id", "")
        logger.info(
            f"Executing background removal with model [{model_id}] on image size {image.size}"
        )

        image_rgba = image.convert("RGBA")

        # 1. Real ONNX inference via rembg session
        if self.rembg_session is not None:
            from rembg import remove
            result = remove(image_rgba, session=self.rembg_session)
            if result.mode != "RGBA":
                result = result.convert("RGBA")
            elapsed = time.time() - start_time
            logger.info(f"rembg [{self.rembg_name}] inference completed in {elapsed:.3f}s")
            return result

        # 2. Built-in lightweight algorithmic fallback (when weights are missing)
        np_img = np.array(image_rgba)
        rgb = np_img[:, :, :3]

        h, w = rgb.shape[:2]
        corners = [
            rgb[0:15, 0:15],
            rgb[0:15, w - 15:w],
            rgb[h - 15:h, 0:15],
            rgb[h - 15:h, w - 15:w],
        ]
        bg_sample = np.mean(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

        diff = np.linalg.norm(rgb.astype(np.float32) - bg_sample.astype(np.float32), axis=2)
        alpha = np.clip((diff - 25) * 6, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=1.5))

        image_rgba.putalpha(alpha_img)
        elapsed = time.time() - start_time
        logger.info(f"Fallback background removal completed in {elapsed:.3f}s")
        return image_rgba
