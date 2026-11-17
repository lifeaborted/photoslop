import io
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from backend.app.config import MODELS_DIR
from backend.app.logger import logger
from backend.app.pipelines.base import BasePipeline


class BackgroundRemovalPipeline(BasePipeline):
    """
    Pipeline for AI background removal supporting:
    - BRIA RMBG-2.0 (SOTA BiRefNet architecture)
    - BRIA RMBG-1.4 (ONNX / CPU Fallback)
    - rembg (U2-Net / IS-Net)
    - Algorithmic fallback when weights are offline
    """

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.model = None
        self.transform = None

    def load(self) -> bool:
        model_id = self.model_info.get("id", "")
        model_name = self.model_info.get("name", "")
        logger.info(f"Loading background removal model: {model_name} [{model_id}]")

        # 1. Try loading BRIA RMBG-2.0 / BiRefNet
        if "rmbg-2.0" in model_id or "birefnet" in model_id:
            try:
                import torch
                from transformers import AutoModelForImageSegmentation
                from torchvision import transforms

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32

                local_path = MODELS_DIR / "RMBG-2.0"
                model_target = str(local_path) if local_path.exists() else "briaai/RMBG-2.0"

                logger.info(f"Initializing RMBG-2.0 on {device} ({dtype})...")
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    model_target,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
                self.model.to(device)
                self.model.eval()

                self.transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                self.is_loaded = True
                logger.info(f"RMBG-2.0 successfully loaded on {device}")
                return True
            except Exception as e:
                logger.warning(f"Could not load RMBG-2.0 weights ({e}). Operating with fallback engine.")

        # 2. Try loading ONNX model (RMBG 1.4 or U2-Net)
        filename = self.model_info.get("filename")
        if filename:
            onnx_path = MODELS_DIR / filename
            if onnx_path.exists():
                try:
                    import onnxruntime as ort
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.model = ort.InferenceSession(str(onnx_path), providers=providers)
                    self.is_loaded = True
                    logger.info(f"ONNX background removal session loaded from {onnx_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load ONNX session from {onnx_path}: {e}")

        # 3. Fallback ready
        self.is_loaded = True
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading background removal model: {self.model_info.get('name')}")
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.is_loaded = False
        return True

    def process(self, image: Image.Image, threshold: int = 128) -> Image.Image:
        """Removes background from PIL Image and returns RGBA Image."""
        start_time = time.time()
        model_id = self.model_info.get("id", "")
        logger.info(
            f"Executing background removal with model [{model_id}] on image size {image.size}"
        )

        image_rgba = image.convert("RGBA")
        orig_w, orig_h = image.size

        # A. Inference via RMBG-2.0 PyTorch
        if self.model is not None and hasattr(self.model, "forward"):
            try:
                import torch
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype

                img_rgb = image.convert("RGB")
                input_tensor = self.transform(img_rgb).unsqueeze(0).to(device=device, dtype=dtype)

                with torch.no_grad():
                    preds = self.model(input_tensor)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()

                # Convert mask to PIL and resize back to original
                from torchvision.transforms.functional import to_pil_image
                mask = to_pil_image(pred).resize((orig_w, orig_h), Image.Resampling.BILINEAR)

                image_rgba.putalpha(mask)
                elapsed = time.time() - start_time
                logger.info(f"RMBG-2.0 inference completed in {elapsed:.3f}s")
                return image_rgba
            except Exception as e:
                logger.error(f"RMBG-2.0 inference failed: {e}. Falling back to default pipeline.")

        # B. Native rembg library check
        try:
            u2net_path = Path.home() / ".u2net" / "u2net.onnx"
            if u2net_path.exists() or model_id == "rembg-u2net":
                import rembg
                result = rembg.remove(image_rgba)
                elapsed = time.time() - start_time
                logger.info(f"rembg successfully removed background in {elapsed:.3f}s")
                return result
        except Exception:
            pass

        # C. Built-in lightweight algorithmic fallback
        np_img = np.array(image_rgba)
        rgb = np_img[:, :, :3]

        h, w = rgb.shape[:2]
        corners = [
            rgb[0:15, 0:15],
            rgb[0:15, w-15:w],
            rgb[h-15:h, 0:15],
            rgb[h-15:h, w-15:w]
        ]
        bg_sample = np.mean(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

        diff = np.linalg.norm(rgb.astype(np.float32) - bg_sample.astype(np.float32), axis=2)
        alpha = np.clip((diff - 25) * 6, 0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=1.5))

        image_rgba.putalpha(alpha_img)
        elapsed = time.time() - start_time
        logger.info(f"Built-in background removal completed in {elapsed:.3f}s")
        return image_rgba
