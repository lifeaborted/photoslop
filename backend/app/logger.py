import sys
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from loguru import logger
from backend.app.config import LOG_FILE_PATH, LOGS_DIR

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Remove default loguru handler
logger.remove()

# Console logger with rich colored format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File logger writing strictly to /logs/log.txt with clean formatting
logger.add(
    str(LOG_FILE_PATH),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    rotation="20 MB",
    retention="14 days",
    encoding="utf-8",
    backtrace=True,
    diagnose=True,
    enqueue=True,  # thread-safe async writing
)


class InterceptHandler(logging.Handler):
    """Intercept standard library logging messages and redirect them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Ignore noisy routine uvicorn access logs from writing to the file log
        if record.name == "uvicorn.access":
            msg = record.getMessage()
            # Suppress static assets and routine polling from the main log
            if any(ext in msg for ext in ("/static/", "favicon.ico", "/api/models/hardware")):
                return

        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Redirect standard library loggers (uvicorn, fastapi) to loguru while silencing access spam."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)

    # Silence uvicorn.access logger so static files and polling don't pollute /logs/log.txt
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = []
    uvicorn_access.propagate = False

    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False

    logger.info(f"Loguru logging initialized. Target log file: {LOG_FILE_PATH}")


def get_memory_telemetry() -> Dict[str, Any]:
    """Retrieves current real-time VRAM (GPU) and RAM memory telemetry."""
    data = {
        "gpu_available": False,
        "vram_used_mb": 0.0,
        "vram_total_mb": 0.0,
        "vram_percent": 0.0,
        "vram_str": "VRAM: N/A (CPU mode)",
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_percent": 0.0,
        "ram_str": "RAM: N/A",
    }

    # 1. System RAM via psutil
    try:
        import psutil
        ram = psutil.virtual_memory()
        data["ram_used_gb"] = round((ram.total - ram.available) / (1024**3), 2)
        data["ram_total_gb"] = round(ram.total / (1024**3), 2)
        data["ram_percent"] = ram.percent
        data["ram_str"] = f"RAM: {data['ram_used_gb']:.2f}GB / {data['ram_total_gb']:.2f}GB ({ram.percent}%)"
    except Exception:
        pass

    # 2. GPU VRAM via PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_bytes = total_bytes - free_bytes
            data["gpu_available"] = True
            data["gpu_name"] = torch.cuda.get_device_name(0)
            data["vram_used_mb"] = round(used_bytes / (1024 * 1024), 1)
            data["vram_total_mb"] = round(total_bytes / (1024 * 1024), 1)
            data["vram_percent"] = round((used_bytes / total_bytes) * 100, 1)
            data["vram_str"] = (
                f"VRAM: {data['vram_used_mb']:.0f}MB / {data['vram_total_mb']:.0f}MB ({data['vram_percent']}%)"
            )
    except Exception:
        pass

    return data


@contextmanager
def track_model_execution(
    category: str,
    model_id: str,
    model_name: str,
    params: Dict[str, Any]
):
    """
    Context manager that monitors model invocation, memory usage, execution speed,
    and records structured success or failure telemetry directly into /logs/log.txt.
    """
    mem_before = get_memory_telemetry()
    param_items = [f"{k}={v}" for k, v in params.items() if v is not None and v != ""]
    param_summary = ", ".join(param_items) if param_items else "default"

    # [1] Log model invocation (Обращение к модели)
    logger.info(
        f"\n>>> [MODEL_CALL] Category: {category} | Model: {model_name} ({model_id})\n"
        f"    Params: {param_summary}\n"
        f"    Memory before: {mem_before['vram_str']} | {mem_before['ram_str']}"
    )

    # Reset PyTorch peak memory stats if CUDA is active
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    start_time = time.perf_counter()
    result_holder = {"output_info": "Completed successfully"}

    class TrackerHandle:
        def set_result(self, desc: str):
            result_holder["output_info"] = desc

    handle = TrackerHandle()

    try:
        yield handle
        elapsed = time.perf_counter() - start_time
        mem_after = get_memory_telemetry()

        # Measure peak memory during inference
        peak_str = ""
        try:
            import torch
            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = round(peak_bytes / (1024 * 1024), 1)
                total_mb = mem_after["vram_total_mb"]
                peak_pct = round((peak_mb / total_mb) * 100, 1) if total_mb > 0 else 0
                peak_str = f"Peak VRAM: {peak_mb:.0f}MB / {total_mb:.0f}MB ({peak_pct}%)"
        except Exception:
            pass

        if not peak_str:
            peak_str = mem_after["vram_str"]

        # Calculate speed
        speed_str = f"{elapsed:.2f}s"
        steps = params.get("steps")
        if steps and isinstance(steps, (int, float)) and steps > 0:
            if elapsed > 0:
                speed_str += f" ({steps / elapsed:.2f} it/s)"

        # [2] Log success outcome (Итог работы: УСПЕХ)
        logger.info(
            f"\n<<< [MODEL_SUCCESS] Category: {category} | Model: {model_id}\n"
            f"    Duration: {speed_str}\n"
            f"    Memory: {peak_str} | {mem_after['ram_str']}\n"
            f"    Result: {result_holder['output_info']}"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        mem_err = get_memory_telemetry()

        # [3] Log error outcome (Итог работы: ОШИБКА)
        logger.error(
            f"\n<<< [MODEL_ERROR] Category: {category} | Model: {model_id}\n"
            f"    Duration: {elapsed:.2f}s\n"
            f"    Error: {type(exc).__name__}: {exc}\n"
            f"    Memory at failure: {mem_err['vram_str']} | {mem_err['ram_str']}"
        )
        raise
