# syntax=docker/dockerfile:1.5

FROM python:3.11.8-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/worker-vision/requirements.worker.cpu.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY deploy/worker-vision/app /app/app
COPY deploy/scripts /app/scripts

ENV REDIS_URL=redis://queue:6379/0 \
    ARTIFACT_DIR=/app/sessions

VOLUME ["/app/sessions"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python /app/scripts/wait_for_redis.py || exit 1

CMD ["python", "-m", "app.main"]


FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common curl ffmpeg libgl1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-distutils \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm -rf /var/lib/apt/lists/* get-pip.py

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY deploy/worker-vision/requirements.worker.cuda.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/requirements.txt

COPY deploy/worker-vision/app /app/app
COPY deploy/scripts /app/scripts

ENV REDIS_URL=redis://queue:6379/0 \
    ARTIFACT_DIR=/app/sessions

VOLUME ["/app/sessions"]
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python /app/scripts/wait_for_redis.py || exit 1

CMD ["python", "-m", "app.main"]
