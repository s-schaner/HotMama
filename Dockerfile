FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip && \
    python tools/install.py --profile cpu

ENV VOLLEYSENSE_HOST=0.0.0.0 \
    VOLLEYSENSE_PORT=8000 \
    VOLLEYSENSE_SESSIONS=/app/sessions \
    VOLLEYSENSE_LOG_DIR=/app/logs

EXPOSE 8000

ENTRYPOINT ["python", "docker/start.py"]
