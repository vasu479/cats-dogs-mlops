# ---------------------------------------------------------------------------
# M2 - Containerised inference service.
# Multi-stage: dependencies are resolved in a builder stage and only the
# resulting virtualenv is copied into a slim runtime image, so compilers and
# pip caches never ship to production.
# ---------------------------------------------------------------------------

# ----------------------------- builder -------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# The extra index makes the pinned torch/torchvision versions resolve to the
# CPU-only Linux wheels (~200 MB) instead of the CUDA build (~2.5 GB).
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ----------------------------- runtime -------------------------------------
FROM python:3.11-slim AS runtime

# Injected by CI (docker build --build-arg GIT_SHA=$GITHUB_SHA) so /health can
# report exactly which commit is serving traffic.
ARG GIT_SHA=local
ARG APP_VERSION=1.0.0

LABEL org.opencontainers.image.title="cats-dogs-api" \
      org.opencontainers.image.description="Cats vs Dogs binary image classifier - MLOps Assignment 2 (AIMLCZG523)" \
      org.opencontainers.image.source="https://github.com/vasu479/cats-dogs-mlops" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/model.pt \
    LOG_LEVEL=INFO \
    GIT_SHA=${GIT_SHA} \
    APP_VERSION=${APP_VERSION}

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Application code, shared library code, config and the trained artifact.
COPY src/ ./src/
COPY app/ ./app/
COPY params.yaml ./params.yaml
COPY models/ ./models/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# stdlib-only healthcheck - avoids installing curl in the runtime image.
# Kept on ONE line: a backslash continuation inside the quoted Python snippet is
# a well-known source of silent healthcheck breakage.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
