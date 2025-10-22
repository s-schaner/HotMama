#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PROFILE=cpu

echo "[HotMama] Launching CPU stack..."
docker compose up -d --build

echo "[HotMama] GUI available at http://localhost:7860 once services are healthy."
echo "[HotMama] Tail logs with: docker compose logs -f"
