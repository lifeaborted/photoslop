import sys
import logging
from loguru import logger
from backend.app.config import LOG_FILE_PATH, LOGS_DIR

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Remove default loguru handler
logger.remove()

# Console logger with rich format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File logger writing strictly to /logs/log.txt
logger.add(
    str(LOG_FILE_PATH),
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
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
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Redirect standard library loggers (uvicorn, fastapi) to loguru."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.INFO)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        mod_logger = logging.getLogger(logger_name)
        mod_logger.handlers = [InterceptHandler()]
        mod_logger.propagate = False

    logger.info(f"Loguru logging initialized. Target log file: {LOG_FILE_PATH}")
