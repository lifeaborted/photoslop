import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.app.models_manager import models_manager
from backend.app.logger import logger
from backend.app.api.editor import data_uri_to_pil, pil_to_data_uri
from backend.app.pipelines.rembg_pipeline import BackgroundRemovalPipeline
from backend.app.pipelines.inpaint_pipeline import InpaintPipeline
from backend.app.pipelines.txt2img_pipeline import TextToImagePipeline

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
    steps: Optional[int] = 4
    seed: Optional[int] = None
    model_id: Optional[str] = None


@router.post("/remove-background")
def remove_background(payload: RemoveBackgroundRequest):
    """Executes AI background removal with the active background model."""
    start_time = time.time()
    active_id = payload.model_id or models_manager.active_models.get("background_removal", "bria-rmbg-1.4")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}

    logger.info(f"API remove-background request using model: {active_id}")
    try:
        input_img = data_uri_to_pil(payload.image_data)
        pipeline = BackgroundRemovalPipeline(model_info)
        result_img = pipeline.process(input_img)

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_info.get("name"),
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

    logger.info(f"API inpaint request. Model: {active_id}, Prompt: '{payload.prompt}'")
    try:
        input_img = data_uri_to_pil(payload.image_data)
        mask_img = data_uri_to_pil(payload.mask_data)

        pipeline = InpaintPipeline(model_info)
        result_img = pipeline.inpaint(
            image=input_img,
            mask=mask_img,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt or "",
            seed=payload.seed,
            steps=payload.steps or 20,
            guidance_scale=payload.guidance_scale or 7.5
        )

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_info.get("name"),
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

    logger.info(f"API text-edit request. Model: {active_id}, Prompt: '{payload.prompt}'")
    try:
        input_img = data_uri_to_pil(payload.image_data)
        pipeline = InpaintPipeline(model_info)
        result_img = pipeline.text_edit(
            image=input_img,
            prompt=payload.prompt,
            strength=payload.strength or 0.5,
            seed=payload.seed
        )

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_info.get("name"),
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
    """Generates a new image from scratch based on text prompt."""
    start_time = time.time()
    active_id = payload.model_id or models_manager.active_models.get("txt2img", "sdxl-turbo-int8")
    model_info = models_manager.get_model_by_id(active_id) or {"id": active_id, "name": active_id}

    logger.info(f"API txt2img request. Model: {active_id}, Prompt: '{payload.prompt}'")
    try:
        pipeline = TextToImagePipeline(model_info)
        result_img = pipeline.generate(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt or "",
            width=payload.width or 512,
            height=payload.height or 512,
            steps=payload.steps or 4,
            seed=payload.seed
        )

        result_uri = pil_to_data_uri(result_img, format="PNG")
        elapsed = round(time.time() - start_time, 3)

        return {
            "status": "success",
            "model_id": active_id,
            "model_name": model_info.get("name"),
            "elapsed_seconds": elapsed,
            "width": result_img.width,
            "height": result_img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error during txt2img: {e}")
        raise HTTPException(status_code=500, detail=str(e))
