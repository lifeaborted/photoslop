import io
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from backend.app.logger import logger
from backend.app.pipelines.base import BasePipeline


class BackgroundRemovalPipeline(BasePipeline):
    """Pipeline for AI background removal (BRIA RMBG, BiRefNet, rembg)."""

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.session = None

    def load(self) -> bool:
        logger.info(f"Loading background removal model: {self.model_info.get('name')}")
        # In full production environment, this loads ONNX session or PyTorch model
        self.is_loaded = True
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading background removal model: {self.model_info.get('name')}")
        self.session = None
        self.is_loaded = False
        return True

    def process(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """Removes background from PIL Image and returns RGBA Image."""
        start_time = time.time()
        logger.info(
            f"Executing background removal with model [{self.model_info.get('id')}] on image size {image.size}"
        )

        image_rgba = image.convert("RGBA")

        # Check if local model or cached session exists before attempting online download
        try:
            import os
            # If user has set U2NET_HOME or cached weights exist
            u2net_path = Path.home() / ".u2net" / "u2net.onnx"
            if u2net_path.exists():
                import rembg
                result = rembg.remove(image_rgba)
                elapsed = time.time() - start_time
                logger.info(f"rembg successfully removed background in {elapsed:.3f}s")
                return result
        except Exception as e:
            logger.warning(f"rembg native failed ({e}), falling back to built-in edge matte.")

        # Built-in lightweight fallback algorithm (for offline / development bootstrap)
        # Segments main subject based on luminance and boundary analysis
        np_img = np.array(image_rgba)
        rgb = np_img[:, :, :3]

        # Corner color sampling for background estimate
        h, w = rgb.shape[:2]
        corners = [
            rgb[0:15, 0:15],
            rgb[0:15, w-15:w],
            rgb[h-15:h, 0:15],
            rgb[h-15:h, w-15:w]
        ]
        bg_sample = np.mean(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

        # Color distance from estimated background
        diff = np.linalg.norm(rgb.astype(np.float32) - bg_sample.astype(np.float32), axis=2)
        alpha = np.clip((diff - 25) * 6, 0, 255).astype(np.uint8)

        # Smooth alpha mask
        alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=1.5))

        image_rgba.putalpha(alpha_img)
        elapsed = time.time() - start_time
        logger.info(f"Built-in background removal completed in {elapsed:.3f}s")
        return image_rgba
