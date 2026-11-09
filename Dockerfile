# syntax=docker/dockerfile:1
#
# FraudGuard API image.
#
# Build from the repository root, after the model artifact exists:
#
#   docker build -t fraudguard-backend .
#
# The artifact (artifacts/models/fraud_model.ubj) is deliberately git-ignored
# by the project, so it must be present in the working tree when this builds.
# Only that one artifact is copied; the EDA/benchmark outputs around it are not.

FROM python:3.13-slim-bookworm

# PYTHONDONTWRITEBYTECODE: no .pyc litter in the image.
# PYTHONUNBUFFERED:      logs reach `docker logs` immediately.
# PIP_NO_CACHE_DIR:      keep the pip cache out of the layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is the OpenMP runtime the XGBoost wheels link against; the slim base
# image does not include it. Nothing else from the OS is needed: psycopg ships
# its own libpq in the binary wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits reuse this layer. requirements.txt is used
# unchanged: it carries the training/EDA/test stacks as well as the API's, which
# makes the image large (see the README); splitting it is a project decision, not
# something this image should quietly do.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Application code plus the two things the running service needs from the repo:
# the schema script (used by the documented SQLite setup) and the model bundle.
# .dockerignore keeps data/, notebooks/, .venv/, node_modules/ and .env out.
COPY app ./app
COPY scripts/init_db.py ./scripts/init_db.py
COPY artifacts/models/fraud_model.ubj ./artifacts/models/fraud_model.ubj

# Non-root runtime. The app only reads the artifact and serves HTTP; it never
# writes to the filesystem (alerts go to the database).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

# /health answers 200 only when the model is resident, so an image that lost its
# artifact is reported unhealthy instead of silently serving 503s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# --proxy-headers lets a reverse proxy in front of the API report the real client
# scheme/host; --forwarded-allow-ips is restricted to private networks. One worker
# on purpose: /metrics-lite counters live in process memory.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"]
