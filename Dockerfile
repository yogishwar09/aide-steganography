# AIDE web console + REST API, served by gunicorn.
#
#   docker build -t aide .
#   docker run --rm -p 5050:5050 -v "$PWD/data:/app/data" aide
#
# Trained models live in ./data/models (gitignored, produced by `aide train`
# and `aide train-detector`); mount ./data so the container uses them.
# Without them the app falls back to the formula-only decision engine and
# skips the ML detector. Set AIDE_API_KEY to require an X-API-Key header.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5050

# libgl/libglib: runtime libraries opencv-headless still links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY aide ./aide
RUN pip install ".[web]"

COPY webapp ./webapp
COPY reports ./reports
RUN mkdir -p data/models && useradd --create-home --uid 10001 aide && chown -R aide /app
USER aide

EXPOSE 5050
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5050/healthz').status == 200 else 1)"

CMD ["sh", "-c", "gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} webapp.app:app"]
