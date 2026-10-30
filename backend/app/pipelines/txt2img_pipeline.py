import time
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from backend.app.logger import logger
from backend.app.pipelines.base import BasePipeline


class TextToImagePipeline(BasePipeline):
    """Pipeline for text-to-image generation (SDXL Turbo, SD 1.5, FLUX)."""

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)

    def load(self) -> bool:
        logger.info(f"Loading txt2img model: {self.model_info.get('name')}")
        self.is_loaded = True
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading txt2img model: {self.model_info.get('name')}")
        self.is_loaded = False
        return True

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        seed: Optional[int] = None
    ) -> Image.Image:
        start_time = time.time()
        logger.info(
            f"Txt2Img generating: '{prompt}' [{width}x{height}, steps={steps}, seed={seed}] "
            f"using [{self.model_info.get('id')}]"
        )

        # In production with PyTorch/Diffusers:
        # output = self.pipe(prompt=prompt, width=width, height=height, num_inference_steps=steps).images[0]

        # Built-in generator fallback with visual abstract gradient art based on prompt seed
        if seed is None:
            seed = int(time.time() * 1000) % 1000000

        np.random.seed(seed)
        c1 = np.random.randint(40, 240, size=3)
        c2 = np.random.randint(40, 240, size=3)

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        r = (c1[0] * (1 - xx) + c2[0] * xx) * (1 - yy) + (c2[0] * (1 - xx) + c1[0] * xx) * yy
        g = (c1[1] * (1 - yy) + c2[1] * yy) * (1 - xx) + (c2[1] * (1 - yy) + c1[1] * yy) * xx
        b = (c1[2] * xx + c2[2] * (1 - xx)) * yy + (c2[2] * xx + c1[2] * (1 - xx)) * (1 - yy)

        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        img = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=10))

        # Render decorative prompt overlay
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, height - 70, width - 20, height - 20], fill=(0, 0, 0, 180))
        draw.text((30, height - 55), f"Generated: {prompt[:40]}...", fill=(255, 255, 255))
        draw.text((30, height - 38), f"Model: {self.model_info.get('name')} | Seed: {seed}", fill=(200, 200, 200))

        elapsed = time.time() - start_time
        logger.info(f"Txt2Img finished in {elapsed:.3f}s")
        return img
