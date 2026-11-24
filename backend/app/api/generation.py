import time
from typing import Any, Dict, Optional, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageColor
from backend.app.models_manager import models_manager
from backend.app.pipeline_manager import pipeline_manager
from backend.app.logger import logger, track_model_execution
from backend.app.api.editor import data_uri_to_pil, pil_to_data_uri

router = APIRouter(prefix="/api/generation", tags=["Generation & AI"])


def _normalize_seed(seed: Optional[int]) -> Optional[int]:
    """Treats negative UI sentinel seeds (e.g. -1) as random (None)."""
    return seed if (seed is not None and seed >= 0) else None


def _resolve_model(
    category: str,
    requested_id: Optional[str],
    fallback_default: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolves a compatible model ID for the requested task category.
    Falls back to the active category model if UI sends a model ID from another category
    (for instance when a shared dropdown mixes `inpaint` and `edit` models).
    """
    default_active = models_manager.active_models.get(category, fallback_default)
    candidate_id = requested_id or default_active
    info = models_manager.get_model_by_id(candidate_id)

    if info and info.get("category") and info["category"] != category:
        logger.warning(
            f"Requested model '{candidate_id}' belongs to category '{info['category']}', "
            f"expected '{category}'. Falling back to active '{category}' model '{default_active}'."
        )
        candidate_id = default_active
        info = models_manager.get_model_by_id(candidate_id)

    resolved_info = info or {"id": candidate_id, "name": candidate_id}
    return candidate_id, resolved_info


class RemoveBackgroundRequest(BaseModel):
    image_data: str
    model_id: Optional[str] = None


class RemoveObjectRequest(BaseModel):
    image_data: str
    mask_data: str
    model_id: Optional[str] = None


class InpaintRequest(BaseModel):
    image_data: str
    mask_data: str
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    model_id: Optional[str] = None


class TextEditRequest(BaseModel):
    image_data: str
    prompt: str
    strength: Optional[float] = 0.5
    seed: Optional[int] = None
    model_id: Optional[str] = None


class ReplaceBackgroundRequest(BaseModel):
    image_data: str
    bg_type: str = "color"  # "color" | "image" | "prompt"
    bg_color: Optional[str] = "#ffffff"
    bg_image_data: Optional[str] = None
    bg_prompt: Optional[str] = None
    bg_model_id: Optional[str] = None
    txt2img_model_id: Optional[str] = None
    seed: Optional[int] = None


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 1
    seed: Optional[int] = None
    model_id: Optional[str] = None


@router.post("/remove-background")
def remove_background(payload: RemoveBackgroundRequest):
    """Executes AI background removal with the active background model."""
    start_time = time.time()
    active_id, model_info = _resolve_model("background_removal", payload.model_id, "birefnet-general-lite")
    model_name = model_info.get("name", active_id)

    try:
        input_img = data_uri_to_pil(payload.image_data)

        with track_model_execution(
            category="background_removal",
            model_id=active_id,
            model_name=model_name,
            params={"size": f"{input_img.width}x{input_img.height}"}
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("background_removal", active_id)
            result_img = pipeline.process(input_img)
            tracker.set_result(f"{result_img.width}x{result_img.height} RGBA")

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_name,
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error in remove-background: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/remove-object")
def remove_object_from_image(payload: RemoveObjectRequest):
    """Erases masked object cleanly without prompt (LaMa FP32 ONNX)."""
    start_time = time.time()
    active_id, model_info = _resolve_model("object_removal", payload.model_id, "lama-cleaner-onnx")
    model_name = model_info.get("name", active_id)

    try:
        input_img = data_uri_to_pil(payload.image_data)
        mask_img = data_uri_to_pil(payload.mask_data)

        with track_model_execution(
            category="object_removal",
            model_id=active_id,
            model_name=model_name,
            params={"size": f"{input_img.width}x{input_img.height}"}
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("object_removal", active_id)
            result_img = pipeline.remove_object(input_img, mask_img)
            tracker.set_result(f"{result_img.width}x{result_img.height} PNG")

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_name,
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error in remove-object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inpaint")
def inpaint_image(payload: InpaintRequest):
    """Executes masked generative inpainting with the active inpaint model (e.g. DreamShaper 8)."""
    start_time = time.time()
    active_id, model_info = _resolve_model("inpaint", payload.model_id, "dreamshaper-8-inpainting")
    model_name = model_info.get("name", active_id)
    seed = _normalize_seed(payload.seed)

    try:
        input_img = data_uri_to_pil(payload.image_data)
        mask_img = data_uri_to_pil(payload.mask_data)

        with track_model_execution(
            category="inpaint",
            model_id=active_id,
            model_name=model_name,
            params={
                "prompt": payload.prompt[:60],
                "steps": payload.steps or 20,
                "guidance": payload.guidance_scale or 7.5,
                "seed": seed,
                "size": f"{input_img.width}x{input_img.height}"
            }
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("inpaint", active_id)
            result_img = pipeline.inpaint(
                image=input_img,
                mask=mask_img,
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                seed=seed,
                steps=payload.steps or 20,
                guidance_scale=payload.guidance_scale or 7.5
            )
            tracker.set_result(f"{result_img.width}x{result_img.height} PNG")

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_name,
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error during inpaint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replace-background")
def replace_background(payload: ReplaceBackgroundRequest):
    """
    Removes background in one click and composites the subject onto:
      - solid hex/named color (bg_type='color')
      - user-supplied background image (bg_type='image')
      - AI-generated background from prompt (bg_type='prompt')
    """
    start_time = time.time()
    bg_id, bg_info = _resolve_model("background_removal", payload.bg_model_id, "birefnet-general-lite")
    gen_id, gen_info = _resolve_model("txt2img", payload.txt2img_model_id, "sdxl-turbo-int8")
    seed = _normalize_seed(payload.seed)
    bg_type = (payload.bg_type or "color").lower().strip()

    try:
        input_img = data_uri_to_pil(payload.image_data)
        target_size = input_img.size

        with track_model_execution(
            category="replace_background",
            model_id=f"{bg_id}+{gen_id}" if bg_type == "prompt" else bg_id,
            model_name=bg_info.get("name", bg_id),
            params={
                "bg_type": bg_type,
                "prompt": (payload.bg_prompt or "")[:60],
                "seed": seed,
                "size": f"{input_img.width}x{input_img.height}",
            },
        ) as tracker:
            # 1. Segment foreground
            bg_pipe = pipeline_manager.get_pipeline("background_removal", bg_id)
            fg_rgba = bg_pipe.process(input_img).convert("RGBA")

            # 2. Build replacement background
            if bg_type == "image" and payload.bg_image_data:
                bg_layer = data_uri_to_pil(payload.bg_image_data).convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
            elif bg_type == "prompt" and payload.bg_prompt:
                txt_pipe = pipeline_manager.get_pipeline("txt2img", gen_id)
                txt_steps = int(gen_info.get("default_steps", 1))
                gen_img = txt_pipe.generate(prompt=payload.bg_prompt, width=512, height=512, steps=txt_steps, seed=seed)
                bg_layer = gen_img.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
            else:
                rgb = ImageColor.getrgb(payload.bg_color or "#ffffff")
                bg_layer = Image.new("RGBA", target_size, (rgb[0], rgb[1], rgb[2], 255))

            composited = Image.alpha_composite(bg_layer, fg_rgba)
            tracker.set_result(f"{composited.width}x{composited.height} RGBA ({bg_type})")

        result_uri = pil_to_data_uri(composited, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "bg_type": bg_type,
            "bg_model_id": bg_id,
            "txt2img_model_id": gen_id if bg_type == "prompt" else None,
            "elapsed_seconds": elapsed,
            "width": composited.width,
            "height": composited.height,
            "data_uri": result_uri,
        }
    except Exception as e:
        logger.error(f"Error in replace-background: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text-edit")
def text_edit_image(payload: TextEditRequest):
    """Executes text-guided whole-image modification (InstructPix2Pix)."""
    start_time = time.time()
    active_id, model_info = _resolve_model("edit", payload.model_id, "instruct-pix2pix-fp16")
    model_name = model_info.get("name", active_id)
    seed = _normalize_seed(payload.seed)

    try:
        input_img = data_uri_to_pil(payload.image_data)

        with track_model_execution(
            category="edit",
            model_id=active_id,
            model_name=model_name,
            params={
                "prompt": payload.prompt[:60],
                "strength": payload.strength or 0.5,
                "seed": seed,
                "size": f"{input_img.width}x{input_img.height}"
            }
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("edit", active_id)
            result_img = pipeline.text_edit(
                image=input_img,
                prompt=payload.prompt,
                strength=payload.strength or 0.5,
                seed=seed
            )
            tracker.set_result(f"{result_img.width}x{result_img.height} PNG")

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_name,
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error during text edit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/txt2img")
def text_to_image(payload: TextToImageRequest):
    """Generates a new image from scratch based on text prompt with real model inference."""
    start_time = time.time()
    active_id, model_info = _resolve_model("txt2img", payload.model_id, "sdxl-turbo-int8")
    model_name = model_info.get("name", active_id)
    seed = _normalize_seed(payload.seed)

    steps = payload.steps if (payload.steps and payload.steps > 0) else 1
    width = payload.width or 512
    height = payload.height or 512

    try:
        with track_model_execution(
            category="txt2img",
            model_id=active_id,
            model_name=model_name,
            params={
                "prompt": payload.prompt[:60],
                "steps": steps,
                "seed": seed,
                "size": f"{width}x{height}"
            }
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("txt2img", active_id)
            result_img = pipeline.generate(
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                width=width,
                height=height,
                steps=steps,
                seed=seed
            )
            tracker.set_result(f"{result_img.width}x{result_img.height} PNG")

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_name,
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error during txt2img: {e}")
        raise HTTPException(status_code=500, detail=str(e))
