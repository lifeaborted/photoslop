from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.app.models_manager import models_manager
from backend.app.logger import logger

router = APIRouter(prefix="/api/models", tags=["Models"])


class SwitchModelRequest(BaseModel):
    category: str
    model_id: str


@router.get("")
def list_models(category: Optional[str] = None):
    """Returns list of models configured in /models/models.json."""
    logger.debug(f"API request: list_models(category={category})")
    models = models_manager.get_models(category)
    active = models_manager.get_active_models()
    return {
        "status": "success",
        "models": models,
        "active_models": active,
        "total_count": len(models)
    }


@router.get("/active")
def get_active():
    """Returns currently active models for each processing category."""
    return {
        "status": "success",
        "active_models": models_manager.get_active_models()
    }


@router.post("/switch")
def switch_model(payload: SwitchModelRequest):
    """Dynamically switches active model for specified category."""
    logger.info(f"Received request to switch model for [{payload.category}] -> {payload.model_id}")
    try:
        result = models_manager.switch_model(payload.category, payload.model_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error switching model")


@router.get("/hardware")
def get_hardware():
    """Returns real-time GPU/VRAM telemetry."""
    return models_manager.get_hardware_status()
