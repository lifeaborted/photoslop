import gc
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from backend.app.logger import logger
from backend.app.config import MODELS_DIR
from backend.app.pipelines.base import BasePipeline

_TORCHAO_PATCHED = False


def _apply_torchao_patches():
    """Applies compatibility patches for TorchAO, PyTorch 2.6+, and GTX 1660 Ti (Turing)."""
    global _TORCHAO_PATCHED
    if _TORCHAO_PATCHED:
        return
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

        _TORCHAO_PATCHED = True
        logger.info("TorchAO compatibility patches & FP32 accumulation applied successfully")
    except Exception as e:
        logger.warning(f"Could not apply TorchAO patches: {e}")


class TextToImagePipeline(BasePipeline):
    """Pipeline for text-to-image generation (SDXL Turbo INT8, SD-Turbo FP16, Segmind SSD-1B)."""

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.pipe = None
        self.device = "cpu"

    def load(self) -> bool:
        model_id = self.model_info.get("id", "")
        filename = self.model_info.get("filename", "")
        model_name = self.model_info.get("name", "")
        logger.info(f"Loading txt2img model: {model_name} [{model_id}]")

        model_path = MODELS_DIR / filename if filename else MODELS_DIR / model_id

        if model_path.exists() and (model_path / "model_index.json").exists():
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available; txt2img will fall back to procedural mode.")
                    self.using_fallback = True
                    self.is_loaded = True
                    return True

                is_int8 = "int8" in model_id or (model_path / "quantization_info.json").exists()
                if is_int8:
                    _apply_torchao_patches()

                from diffusers import AutoPipelineForText2Image

                idx_data = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
                class_name = idx_data.get("_class_name", "")

                # Auto-detect fp16 variant files
                has_fp16 = (model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()
                variant = self.model_info.get("variant") or ("fp16" if has_fp16 else None)
                if variant == "fp16" and not has_fp16:
                    variant = None

                load_kwargs: Dict[str, Any] = {
                    "torch_dtype": torch.float16,
                    "variant": variant,
                }
                if "safety_checker" in idx_data:
                    load_kwargs["safety_checker"] = None
                    load_kwargs["requires_safety_checker"] = False

                logger.info(f"Loading Diffusers txt2img from: {model_path} (variant={variant}, class={class_name})")
                pipe = AutoPipelineForText2Image.from_pretrained(str(model_path), **load_kwargs)

                # Memory optimizations for 6GB VRAM
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()

                # VAE FP16 overflow guard for ALL Diffusers models on Turing GPUs (GTX 1660 Ti)
                if hasattr(pipe, "vae"):
                    pipe.vae.to(torch.float32)
                    orig_decode = pipe.vae.decode
                    pipe.vae.decode = lambda z, *a, **kw: orig_decode(z.to(torch.float32), *a, **kw)

                try:
                    pipe.to("cuda")
                    self.device = "cuda"
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM placing [{model_id}] fully on CUDA; switching to enable_model_cpu_offload()")
                    torch.cuda.empty_cache()
                    pipe.enable_model_cpu_offload()
                    self.device = "cuda"

                self.pipe = pipe
                self.using_fallback = False
                self.is_loaded = True
                logger.info(f"Diffusers txt2img pipeline [{model_id}] loaded on {torch.cuda.get_device_name(0)}")
                return True
            except Exception as e:
                logger.error(f"Failed to load Diffusers model from {model_path}: {e}")
                raise

        # Weights not present locally -> explicit fallback flag
        self.using_fallback = True
        self.is_loaded = True
        logger.warning(f"Weights not found at {model_path}; txt2img [{model_id}] running in FALLBACK mode.")
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading txt2img model: {self.model_info.get('name')}")
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
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

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        start_time = time.time()
        model_id = self.model_info.get("id", "")

        eff_steps = steps if (steps is not None and steps > 0) else int(self.model_info.get("default_steps", 1))
        if guidance_scale is None:
            default_g = 0.0 if "turbo" in model_id else 7.5
            eff_guidance = float(self.model_info.get("default_guidance", default_g))
        else:
            eff_guidance = float(guidance_scale)

        logger.info(
            f"Txt2Img generating: '{prompt[:60]}' [{width}x{height}, steps={eff_steps}, "
            f"guidance={eff_guidance}, seed={seed}] using [{model_id}]"
        )

        if self.pipe is not None:
            import torch
            generator = torch.Generator(device="cuda")
            if seed is not None:
                generator.manual_seed(seed)

            w_div = max(64, (width // 8) * 8)
            h_div = max(64, (height // 8) * 8)

            call_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "width": w_div,
                "height": h_div,
                "num_inference_steps": max(1, eff_steps),
                "guidance_scale": eff_guidance,
                "generator": generator,
            }
            if negative_prompt and eff_guidance > 1.0:
                call_kwargs["negative_prompt"] = negative_prompt

            image = self.pipe(**call_kwargs).images[0]

            elapsed = time.time() - start_time
            logger.info(f"Txt2Img Diffusers generation completed in {elapsed:.3f}s")
            return image

        # Built-in fallback when weights are not downloaded
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

        draw = ImageDraw.Draw(img)
        draw.rectangle([20, height - 70, width - 20, height - 20], fill=(0, 0, 0, 180))
        draw.text((30, height - 55), f"Generated: {prompt[:40]}...", fill=(255, 255, 255))
        draw.text((30, height - 38), f"Model: {self.model_info.get('name')} | Seed: {seed}", fill=(200, 200, 200))

        elapsed = time.time() - start_time
        logger.info(f"Txt2Img procedural fallback finished in {elapsed:.3f}s")
        return img
