FROM python:3.11-slim

# Prevent Python from writing .pyc and enable bufferless output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (build-essential, git, libgl1 for OpenCV/Pillow, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime directories
RUN mkdir -p /app/models /app/output /app/logs /app/frontend

# Copy source code
COPY backend /app/backend
COPY frontend /app/frontend

ENV PROJECT_ROOT=/app \
    MODELS_DIR=/app/models \
    OUTPUT_DIR=/app/output \
    LOGS_DIR=/app/logs \
    FRONTEND_DIR=/app/frontend \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
