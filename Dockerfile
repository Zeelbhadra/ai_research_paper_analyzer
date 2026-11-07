# syntax=docker/dockerfile:1
# PaperLens: Flask + retrieval/RAG backend. CPU-only image (no CUDA).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    STORAGE_DIR=/app/storage \
    DATA_DIR=/app/data \
    PORT=5000

WORKDIR /app

# libgomp: OpenMP runtime used by faiss/torch; curl: container health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached layer). CPU-only torch wheel is far smaller than the default CUDA build.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY app.py ./
COPY src ./src
COPY backend ./backend
COPY templates ./templates
COPY static ./static

# Non-root user; storage/data/model-cache are mounted volumes (nothing mutable is baked into the image, no secrets either)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/storage /app/data /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
VOLUME ["/app/storage", "/app/.cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# One worker on purpose: models are loaded once (memory) and the optional scheduler must run exactly once.
# Concurrency comes from threads (retrieval runs outside the service lock).
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 8 --timeout 300 --graceful-timeout 30 --access-logfile - --bind 0.0.0.0:${PORT} app:app"]
