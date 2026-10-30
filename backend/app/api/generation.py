import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.models_manager import models_manager
from backend.app.pipeline_manager import pipeline_manager
from backend.app.logger import logger, track_model_execution
from backend.app.api.editor import data_uri_to_pil, pil_to_data_uri

router = APIRouter(prefix="/api/generation", tags=["Generation & AI"])


class RemoveBackgroundRequest(BaseModel):
    image_data: str
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
    active_id = payload.model_id or models_manager.active_models.get("background_removal", "bria-rmbg-1.4")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}
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


@router.post("/inpaint")
def inpaint_image(payload: InpaintRequest):
    """Executes inpainting with mask using active inpaint model (e.g. FLUX Fill or SD 1.5)."""
    start_time = time.time()
    active_id = payload.model_id or models_manager.active_models.get("inpaint", "sd15-inpainting-fp16")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}
    model_name = model_info.get("name", active_id)

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
                "seed": payload.seed,
                "size": f"{input_img.width}x{input_img.height}"
            }
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("inpaint", active_id)
            result_img = pipeline.inpaint(
                image=input_img,
                mask=mask_img,
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt or "",
                seed=payload.seed,
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


@router.post("/text-edit")
def text_edit_image(payload: TextEditRequest):
    """Executes text-guided image modification (e.g. FLUX Kontext / InstructPix2Pix)."""
    start_time = time.time()
    active_id = payload.model_id or models_manager.active_models.get("edit", "flux1-kontext-q2")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}
    model_name = model_info.get("name", active_id)

    try:
        input_img = data_uri_to_pil(payload.image_data)

        with track_model_execution(
            category="edit",
            model_id=active_id,
            model_name=model_name,
            params={
                "prompt": payload.prompt[:60],
                "strength": payload.strength or 0.5,
                "seed": payload.seed,
                "size": f"{input_img.width}x{input_img.height}"
            }
        ) as tracker:
            pipeline = pipeline_manager.get_pipeline("edit", active_id)
            result_img = pipeline.text_edit(
                image=input_img,
                prompt=payload.prompt,
                strength=payload.strength or 0.5,
                seed=payload.seed
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
    active_id = payload.model_id or models_manager.active_models.get("txt2img", "sdxl-turbo-int8")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}
    model_name = model_info.get("name", active_id)

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
                "seed": payload.seed,
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
                seed=payload.seed
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
