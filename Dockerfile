# Chatterbox TTS API Server
# Build: docker build -t chatterbox-tts .
# Run: docker run --gpus all -p 8000:8000 chatterbox-tts

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY server/ ./server/

# Install dependencies
RUN pip3 install --no-cache-dir -e ".[server]"

# Pre-download models (optional, comment out to download at runtime)
# RUN python3 -c "from chatterbox import ChatterboxTurboTTS; ChatterboxTurboTTS.from_pretrained('cpu')"

# Create voices directory
RUN mkdir -p /app/voices && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Run server
CMD ["python3", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
