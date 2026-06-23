# syntax=docker/dockerfile:1

# ---- builder: install pinned runtime deps into an isolated prefix ----
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime: slim, non-root ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY data/corpus/ ./data/corpus/

# Writable state dirs owned by the non-root user.
RUN mkdir -p db cache reports && chown -R appuser:appuser /app
USER appuser

ENV DATA_ROOT=data/corpus \
    DB_PATH=db/lancedb \
    CACHE_PATH=cache/diskcache \
    REPORTS_PATH=reports \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else 1)"

# Shell form so Render's $PORT is respected (falls back to 8000 locally).
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
