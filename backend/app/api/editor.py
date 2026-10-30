import io
import os
import base64
import time
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageEnhance
from backend.app.config import OUTPUT_DIR
from backend.app.logger import logger

router = APIRouter(prefix="/api/editor", tags=["Editor"])


class PasteRequest(BaseModel):
    image_data: str  # Base64 data URL
    name: Optional[str] = "Pasted_Image"


class AdjustmentsRequest(BaseModel):
    image_data: str  # Base64 data URL
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    rotate_angle: float = 0.0
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


class SaveImageRequest(BaseModel):
    image_data: str  # Base64 data URL
    filename: Optional[str] = None
    format: str = "PNG"  # PNG or JPEG


def data_uri_to_pil(data_uri: str) -> Image.Image:
    """Converts a base64 data URI (e.g. data:image/png;base64,...) to PIL Image."""
    if "," in data_uri:
        header, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes))


def pil_to_data_uri(img: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image to a base64 data URI."""
    buf = io.BytesIO()
    save_format = "JPEG" if format.upper() in ("JPG", "JPEG") else "PNG"
    if save_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=save_format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if save_format == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Accepts single or multiple files (batch upload).
    Returns list of processed layer objects ready for frontend canvas layers.
    """
    logger.info(f"Received batch upload of {len(files)} file(s).")
    layers = []

    for idx, file in enumerate(files):
        try:
            content = await file.read()
            pil_img = Image.open(io.BytesIO(content))
            width, height = pil_img.size
            format_name = pil_img.format or "PNG"

            # Re-encode to data URI for frontend canvas consumption
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            layer_info = {
                "id": f"layer_{int(time.time() * 1000)}_{idx}",
                "name": file.filename or f"Layer_{idx + 1}",
                "width": width,
                "height": height,
                "data_uri": data_uri,
                "format": format_name,
                "size_bytes": len(content)
            }
            layers.append(layer_info)
            logger.debug(f"Processed uploaded layer: {layer_info['name']} ({width}x{height})")
        except Exception as e:
            logger.error(f"Error parsing uploaded file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

    logger.info(f"Successfully processed {len(layers)} image layer(s) for frontend distribution.")
    return {
        "status": "success",
        "count": len(layers),
        "layers": layers
    }


@router.post("/paste")
def paste_image(payload: PasteRequest):
    """
    Receives image data pasted from clipboard (Ctrl+V) as base64 data URI.
    """
    logger.info(f"Processing clipboard paste request: name='{payload.name}'")
    try:
        pil_img = data_uri_to_pil(payload.image_data)
        width, height = pil_img.size

        layer_info = {
            "id": f"layer_paste_{int(time.time() * 1000)}",
            "name": payload.name or f"Pasted_{int(time.time())}",
            "width": width,
            "height": height,
            "data_uri": payload.image_data
        }
        logger.info(f"Pasted image registered: {width}x{height}")
        return {
            "status": "success",
            "layer": layer_info
        }
    except Exception as e:
        logger.error(f"Failed to process pasted clipboard image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse pasted image: {e}")


@router.post("/adjustments")
def apply_adjustments(payload: AdjustmentsRequest):
    """Applies basic adjustments: brightness, contrast, saturation, rotation, resize."""
    try:
        img = data_uri_to_pil(payload.image_data)

        # Brightness
        if payload.brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(payload.brightness)

        # Contrast
        if payload.contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(payload.contrast)

        # Saturation (Color)
        if payload.saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(payload.saturation)

        # Rotate
        if payload.rotate_angle != 0.0:
            img = img.rotate(payload.rotate_angle, expand=True)

        # Resize
        if payload.resize_width and payload.resize_height:
            img = img.resize((payload.resize_width, payload.resize_height), Image.Resampling.LANCZOS)

        result_uri = pil_to_data_uri(img, format="PNG")
        return {
            "status": "success",
            "width": img.width,
            "height": img.height,
            "data_uri": result_uri
        }
    except Exception as e:
        logger.error(f"Error during adjustments: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save")
def save_output(payload: SaveImageRequest):
    """Saves image directly to /output directory (PNG / JPG)."""
    try:
        img = data_uri_to_pil(payload.image_data)
        format_ext = payload.format.lower().strip()
        if format_ext not in ("png", "jpg", "jpeg"):
            format_ext = "png"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if payload.filename:
            clean_name = Path(payload.filename).stem
            file_name = f"{clean_name}_{timestamp}.{format_ext}"
        else:
            file_name = f"pepeshop_export_{timestamp}.{format_ext}"

        dest_path = OUTPUT_DIR / file_name

        save_fmt = "JPEG" if format_ext in ("jpg", "jpeg") else "PNG"
        if save_fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")

        img.save(dest_path, format=save_fmt, quality=95)
        file_size_kb = round(dest_path.stat().st_size / 1024, 2)

        logger.info(f"Image successfully saved to /output: {dest_path.name} ({file_size_kb} KB)")

        return {
            "status": "success",
            "filename": file_name,
            "path": str(dest_path),
            "size_kb": file_size_kb,
            "width": img.width,
            "height": img.height
        }
    except Exception as e:
        logger.error(f"Failed to save image to /output: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/outputs")
def list_outputs():
    """Lists all files saved in /output."""
    files = []
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                files.append({
                    "name": p.name,
                    "size_kb": round(p.stat().st_size / 1024, 2),
                    "modified": p.stat().st_mtime
                })
    return {
        "status": "success",
        "output_dir": str(OUTPUT_DIR),
        "files": sorted(files, key=lambda x: x["modified"], reverse=True)
    }
