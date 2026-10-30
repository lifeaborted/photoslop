import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.config import APP_NAME, APP_VERSION, FRONTEND_DIR, LOG_FILE_PATH
from backend.app.logger import logger, setup_logging
from backend.app.models_manager import models_manager
from backend.app.pipeline_manager import pipeline_manager
from backend.app.api.models import router as models_router
from backend.app.api.editor import router as editor_router
from backend.app.api.generation import router as generation_router

# Initialize custom Loguru configuration
setup_logging()

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Local Photoshop-like AI Image Editor with Dynamic Model Switching & Multi-Layer Canvas"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware with Loguru (filtered to eliminate routine noise)
@app.middleware("http")
async def loguru_requests_middleware(request: Request, call_next):
    start_time = time.time()
    path = request.url.path

    # Filter out noisy routine requests from polluting the log file
    is_noisy = (
        path.startswith("/static")
        or path.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".ico", ".woff2"))
        or path in ("/", "/favicon.ico", "/api/models/hardware", "/api/models", "/api/models/active")
    )

    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    # Only log significant actions or errors
    if not is_noisy or response.status_code >= 400:
        logger.info(
            f"HTTP {request.method} {path} -> {response.status_code} ({duration_ms:.1f}ms)"
        )

    return response


# Include API routers
app.include_router(models_router)
app.include_router(editor_router)
app.include_router(generation_router)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Logs directed to: {LOG_FILE_PATH}")
    
    # Telemetry report
    hw = models_manager.get_hardware_status()
    if hw["gpu_available"]:
        logger.info(
            f"Hardware detected: {hw['gpu_name']} | Driver: {hw['driver_version']} | "
            f"VRAM: {hw['vram_used_mb']:.0f}MB / {hw['vram_total_mb']:.0f}MB ({hw['vram_percent']}%)"
        )
    else:
        logger.warning("No NVIDIA GPU detected by nvidia-smi. Running on CPU mode.")

    models = models_manager.get_models()
    logger.info(f"Loaded {len(models)} model definitions from models.json")
    for m in models:
        status_str = "READY" if m.get("is_downloaded") else "NOT DOWNLOADED"
        logger.info(f"  - [{m['category']}] {m['id']} ({m['name']}) -> {status_str}")

    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {APP_NAME}. Cleaning up GPU pipelines...")
    pipeline_manager.unload_all()
    logger.info(f"{APP_NAME} shutdown complete. Goodbye!")


# Serve frontend static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found, but API is operational."}
