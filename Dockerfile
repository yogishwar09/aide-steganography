# AIDE web console + REST API, served by gunicorn.
#
#   docker build -t aide .
#   docker run --rm -p 5050:5050 -v "$PWD/data:/app/data" aide
#
# The trained models in ./data/models (from `aide train` and
# `aide train-detector`) are baked into the image; mount ./data to override
# them. Runtime versions are pinned in constraints.txt because the models
# are version-sensitive scikit-learn pickles. Set AIDE_API_KEY to require
# an X-API-Key header on the versioned /api/v1 endpoints.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5050 \
    WEB_CONCURRENCY=2

# libgl/libglib: runtime libraries opencv-headless still links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md constraints.txt ./
COPY aide ./aide
RUN pip install -c constraints.txt ".[web]"

COPY webapp ./webapp
COPY reports ./reports
COPY data/models ./data/models
RUN useradd --create-home --uid 10001 aide && chown -R aide /app
USER aide

EXPOSE 5050
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '5050')).status == 200 else 1)"

# gunicorn reads its worker count from WEB_CONCURRENCY. Each worker peaks at
# ~300 MB during a full demo run, so 512 MB hosts should set it to 1.
CMD ["sh", "-c", "gunicorn --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} webapp.app:app"]
