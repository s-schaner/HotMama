#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PROFILE=gpu

echo "[HotMama] Launching GPU stack (GUI + CUDA worker)..."
docker compose --profile gpu up -d --build

echo "[HotMama] Ensure NVIDIA Container Toolkit is installed on the host."
echo "[HotMama] GUI available at http://localhost:7860 once services are healthy."
echo "[HotMama] Tail logs with: docker compose --profile gpu logs -f"
