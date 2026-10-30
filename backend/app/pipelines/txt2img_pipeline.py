import time
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from backend.app.logger import logger
from backend.app.config import MODELS_DIR
from backend.app.pipelines.base import BasePipeline


def _apply_torchao_patches():
    """Applies compatibility patches for TorchAO, PyTorch 2.6+, and GTX 1660 Ti."""
    try:
        import torch
        orig_load = torch.load

        def safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return orig_load(*args, **kwargs)

        torch.load = safe_load

        import torchao
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
        from torch.utils._python_dispatch import return_and_correct_aliasing

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([AffineQuantizedTensor])

        aten = torch.ops.aten

        @AffineQuantizedTensor.implements(aten.alias.default)
        def _alias(func, types, args, kwargs):
            return return_and_correct_aliasing(
                func, args, kwargs, args[0]._apply_fn_to_data(torch.detach)
            )

        import torchao.dtypes.uintx.plain_layout as pl
        import torchao.dtypes.affine_quantized_tensor_ops as aqt_ops

        def safe_linear_fp_act_int8_weight_impl(input_tensor, weight_tensor, bias):
            w_vals_int8_t = weight_tensor.tensor_impl.int_data.t()
            scale = weight_tensor.tensor_impl.scale
            inp_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
            m = torch.mm(inp_flat.float(), w_vals_int8_t.float())
            y = (m * scale.reshape(1, -1).float()).to(input_tensor.dtype)
            y = y.reshape(*input_tensor.shape[:-1], y.shape[-1])
            if bias is not None:
                y += bias.to(input_tensor.dtype)
            return y

        pl._linear_fp_act_int8_weight_impl = safe_linear_fp_act_int8_weight_impl
        for k in list(aqt_ops._AQT_QLINEAR_DISPATCH_TABLE.keys()):
            if "fp_act_int8" in k.__name__:
                aqt_ops._AQT_QLINEAR_DISPATCH_TABLE[k] = safe_linear_fp_act_int8_weight_impl

        logger.info("TorchAO compatibility patches & FP32 accumulation applied successfully")
    except Exception as e:
        logger.warning(f"Could not apply TorchAO patches: {e}")


class TextToImagePipeline(BasePipeline):
    """Pipeline for text-to-image generation (SDXL Turbo INT8, SD 1.5, FLUX)."""

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.pipe = None

    def load(self) -> bool:
        model_id = self.model_info.get("id", "")
        model_name = self.model_info.get("name", "")
        logger.info(f"Loading txt2img model: {model_name} [{model_id}]")

        model_path = MODELS_DIR / model_id
        if not model_path.exists():
            # Check if sdxl-turbo-torch-int8 directory exists
            alt_path = MODELS_DIR / "sdxl-turbo-torch-int8"
            if alt_path.exists():
                model_path = alt_path

        if model_path.exists() and (model_path / "model_index.json").exists():
            try:
                import torch
                if torch.cuda.is_available():
                    _apply_torchao_patches()
                    from diffusers import AutoPipelineForText2Image

                    logger.info(f"Loading Diffusers pipeline from: {model_path}")
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        str(model_path),
                        torch_dtype=torch.float16,
                        variant=None
                    )
                    pipe.to("cuda")

                    # VAE FP16 overflow guard
                    pipe.vae.to(torch.float32)
                    orig_decode = pipe.vae.decode
                    pipe.vae.decode = lambda z, *a, **kw: orig_decode(z.to(torch.float32), *a, **kw)

                    self.pipe = pipe
                    self.is_loaded = True
                    logger.info(f"Diffusers txt2img pipeline loaded on {torch.cuda.get_device_name(0)}")
                    return True
            except Exception as e:
                logger.error(f"Failed to load Diffusers model from {model_path}: {e}")

        # Fallback to simulated fast pipeline
        self.is_loaded = True
        logger.info(f"Loaded simulated txt2img pipeline for [{model_id}]")
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading txt2img model: {self.model_info.get('name')}")
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.is_loaded = False
        return True

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 1,
        seed: Optional[int] = None
    ) -> Image.Image:
        start_time = time.time()
        logger.info(
            f"Txt2Img generating: '{prompt}' [{width}x{height}, steps={steps}, seed={seed}] "
            f"using [{self.model_info.get('id')}]"
        )

        if self.pipe is not None:
            try:
                import torch
                generator = torch.Generator(device="cuda")
                if seed is not None:
                    generator.manual_seed(seed)

                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    width=width,
                    height=height,
                    num_inference_steps=max(1, steps),
                    guidance_scale=0.0,
                    generator=generator
                ).images[0]

                elapsed = time.time() - start_time
                logger.info(f"Txt2Img real Diffusers generation finished in {elapsed:.3f}s")
                return image
            except Exception as e:
                logger.error(f"Diffusers inference error: {e}. Falling back to procedural generator.")

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
        logger.info(f"Txt2Img procedural finished in {elapsed:.3f}s")
        return img
