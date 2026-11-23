import gc
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from backend.app.config import MODELS_DIR
from backend.app.logger import logger
from backend.app.pipelines.base import BasePipeline


class InpaintPipeline(BasePipeline):
    """
    Pipeline for:
    - Clean object removal without text (LaMa FP32 ONNX — Fast Fourier Convolutions, 512x512)
    - Generative inpainting with text (DreamShaper 8 Inpainting, SD 2.0 Inpainting, SD 1.5 Inpainting — all <= 4 GB VRAM)
    - Lightweight instruction-guided filters fallback
    """

    def __init__(self, model_info: Dict[str, Any]):
        super().__init__(model_info)
        self.pipe = None
        self.session = None

    def load(self) -> bool:
        model_id = self.model_info.get("id", "")
        model_name = self.model_info.get("name", "")
        filename = self.model_info.get("filename", "")
        logger.info(f"Loading inpaint/edit pipeline: {model_name} [{model_id}]")

        # 1. LaMa ONNX model (object_removal)
        if "lama" in model_id:
            candidates = [
                MODELS_DIR / filename if filename else None,
                MODELS_DIR / "lama_fp32.onnx",
                MODELS_DIR / "lama.onnx",
            ]
            onnx_path = next((p for p in candidates if p is not None and p.exists()), None)
            if onnx_path is not None:
                try:
                    import onnxruntime as ort
                    avail = ort.get_available_providers()
                    providers = (
                        ["CUDAExecutionProvider", "CPUExecutionProvider"]
                        if "CUDAExecutionProvider" in avail
                        else ["CPUExecutionProvider"]
                    )
                    self.session = ort.InferenceSession(str(onnx_path), providers=providers)
                    self.using_fallback = False
                    self.is_loaded = True
                    logger.info(f"LaMa ONNX session loaded from {onnx_path} (providers={self.session.get_providers()})")
                    return True
                except Exception as e:
                    logger.error(f"Could not load LaMa ONNX from {onnx_path}: {e}")
                    raise

        # 2. Diffusers Inpainting pipeline (DreamShaper 8 / SD 2.0 / SD 1.5)
        model_path = MODELS_DIR / filename if filename else MODELS_DIR / model_id
        if model_path.is_dir() and (model_path / "model_index.json").exists():
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA is not available; inpaint will use fallback mode.")
                    self.using_fallback = True
                    self.is_loaded = True
                    return True

                from diffusers import AutoPipelineForInpainting

                idx_data = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
                class_name = idx_data.get("_class_name", "")

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

                logger.info(f"Loading Diffusers inpaint from {model_path} (variant={variant}, class={class_name})")
                pipe = AutoPipelineForInpainting.from_pretrained(str(model_path), **load_kwargs)

                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()

                # VAE FP16 overflow guard for ALL Diffusers inpaint models on Turing GPUs (GTX 1660 Ti)
                if hasattr(pipe, "vae"):
                    pipe.vae.to(torch.float32)
                    orig_decode = pipe.vae.decode
                    pipe.vae.decode = lambda z, *a, **kw: orig_decode(z.to(torch.float32), *a, **kw)

                try:
                    pipe.to("cuda")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM placing [{model_id}] on CUDA; switching to enable_model_cpu_offload()")
                    torch.cuda.empty_cache()
                    pipe.enable_model_cpu_offload()

                self.pipe = pipe
                self.using_fallback = False
                self.is_loaded = True
                logger.info(f"Diffusers inpaint pipeline [{model_id}] loaded on {torch.cuda.get_device_name(0)}")
                return True
            except Exception as e:
                logger.error(f"Diffusers inpaint loading failed for {model_path}: {e}")
                raise

        self.using_fallback = True
        self.is_loaded = True
        logger.warning(f"Weights not found for [{model_id}]; inpaint/object_removal operating in FALLBACK mode.")
        return True

    def unload(self) -> bool:
        logger.info(f"Unloading inpaint/edit model: {self.model_info.get('name')}")
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.session is not None:
            del self.session
            self.session = None

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

    def remove_object(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Erases masked object reconstructing background texture (LaMa FP32 ONNX).
        Unmasked pixels are preserved verbatim from the input image.
        """
        start_time = time.time()
        logger.info(f"Object removal triggered (Model: {self.model_info.get('id')})")

        img_rgb = image.convert("RGB")
        orig_size = img_rgb.size
        mask_l = mask.convert("L").resize(orig_size, Image.Resampling.NEAREST)

        # 1. Real LaMa ONNX inference (fixed 512x512 input shape per Carve/LaMa-ONNX spec)
        if self.session is not None:
            img_512 = img_rgb.resize((512, 512), Image.Resampling.LANCZOS)
            mask_512 = mask_l.resize((512, 512), Image.Resampling.NEAREST)

            img_np = np.array(img_512).astype(np.float32) / 255.0
            mask_np = (np.array(mask_512) > 127).astype(np.float32)

            img_tensor = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 512, 512)
            mask_tensor = mask_np[np.newaxis, np.newaxis, ...]              # (1, 1, 512, 512)

            in_names = [inp.name for inp in self.session.get_inputs()]
            ort_inputs = {
                in_names[0]: img_tensor,
                in_names[1]: mask_tensor,
            }
            pred = self.session.run(None, ort_inputs)[0]
            out_hwc = pred[0].transpose(1, 2, 0)
            if float(np.max(out_hwc)) <= 1.5:
                out_hwc = out_hwc * 255.0
            out_np = np.clip(out_hwc, 0, 255).astype(np.uint8)
            pred_img = Image.fromarray(out_np, mode="RGB").resize(orig_size, Image.Resampling.LANCZOS)

            # Composite ONLY the masked area back onto the crisp original image
            bin_mask = mask_l.point(lambda p: 255 if p > 127 else 0)
            result = img_rgb.copy()
            result.paste(pred_img, (0, 0), mask=bin_mask)

            elapsed = time.time() - start_time
            logger.info(f"LaMa ONNX object removal completed in {elapsed:.3f}s")
            return result

        # 2. Content-Aware blur fallback when weights are missing
        out = img_rgb.copy()
        mask_arr = np.array(mask_l)
        if np.any(mask_arr > 128):
            blurred_coarse = out.filter(ImageFilter.GaussianBlur(radius=16))
            blurred_fine = out.filter(ImageFilter.GaussianBlur(radius=4))
            synth = Image.blend(blurred_coarse, blurred_fine, alpha=0.5)
            out.paste(synth, (0, 0), mask=mask_l)

        elapsed = time.time() - start_time
        logger.info(f"Fallback object erasure completed in {elapsed:.3f}s")
        return out

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        steps: int = 20,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """Inpaints the masked region using the specified text prompt."""
        start_time = time.time()
        logger.info(
            f"Starting inpainting with model [{self.model_info.get('id')}]. "
            f"Prompt: '{prompt[:60]}', Steps: {steps}, Seed: {seed}"
        )

        image_rgb = image.convert("RGB")
        orig_size = image_rgb.size
        mask_l = mask.convert("L").resize(orig_size, Image.Resampling.NEAREST)

        # 1. Real Diffusers Inpaint execution
        if self.pipe is not None:
            import torch
            gen = torch.Generator("cuda")
            if seed is not None:
                gen.manual_seed(seed)

            # Standardize inference resolution to 512x512 (native for SD1.5 / SD2 / DreamShaper)
            w_inf = min(512, max(64, (image_rgb.width // 8) * 8))
            h_inf = min(512, max(64, (image_rgb.height // 8) * 8))

            img_res = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                image=image_rgb.resize((w_inf, h_inf), Image.Resampling.LANCZOS),
                mask_image=mask_l.resize((w_inf, h_inf), Image.Resampling.NEAREST),
                width=w_inf,
                height=h_inf,
                num_inference_steps=max(1, steps),
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0].resize(orig_size, Image.Resampling.LANCZOS)

            # Preserve unmasked pixels cleanly
            bin_mask = mask_l.point(lambda p: 255 if p > 127 else 0)
            out = image_rgb.copy()
            out.paste(img_res, (0, 0), mask=bin_mask)

            elapsed = time.time() - start_time
            logger.info(f"Diffusers inpainting finished in {elapsed:.3f}s")
            return out

        # 2. Fallback synthesizer when weights are missing
        out = image_rgb.copy()
        if np.any(np.array(mask_l) > 128):
            blurred_bg = out.filter(ImageFilter.GaussianBlur(radius=8))
            draw = ImageDraw.Draw(blurred_bg)
            w, h = out.size
            draw.text((w // 4, h // 2), f"AI Inpaint: {prompt[:30]}...", fill=(255, 255, 255))
            out.paste(blurred_bg, (0, 0), mask=mask_l)

        elapsed = time.time() - start_time
        logger.info(f"Fallback inpainting finished in {elapsed:.3f}s")
        return out

    def text_edit(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Applies lightweight global image adjustments by instruction."""
        start_time = time.time()
        logger.info(f"Executing text-edit with prompt: '{prompt}' (Model: {self.model_info.get('id')})")

        img = image.convert("RGB")
        prompt_lower = prompt.lower()

        if "vintage" in prompt_lower or "retro" in prompt_lower or "sepia" in prompt_lower:
            img = ImageEnhance.Color(img).enhance(0.5)
            img = ImageEnhance.Contrast(img).enhance(1.2)
        elif "bright" in prompt_lower or "vibrant" in prompt_lower:
            img = ImageEnhance.Color(img).enhance(1.4)
            img = ImageEnhance.Brightness(img).enhance(1.1)
        elif "dark" in prompt_lower or "night" in prompt_lower:
            img = ImageEnhance.Brightness(img).enhance(0.6)
            img = ImageEnhance.Contrast(img).enhance(1.3)
        elif "black and white" in prompt_lower or "monochrome" in prompt_lower:
            img = ImageOps.grayscale(img).convert("RGB")

        elapsed = time.time() - start_time
        logger.info(f"Text-guided edit finished in {elapsed:.3f}s")
        return img
