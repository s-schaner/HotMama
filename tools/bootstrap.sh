#!/usr/bin/env bash
# Bootstrap installer for HotMama.

set -euo pipefail

REPO_URL=${HOTMAMA_REPO:-"https://github.com/s-schaner/HotMama.git"}
INSTALL_DIR=${HOTMAMA_DIR:-"$HOME/HotMama"}
BRANCH=${HOTMAMA_BRANCH:-"main"}
PROFILE_OVERRIDE=${HOTMAMA_PROFILE:-""}

log() {
  printf '[bootstrap] %s\n' "$1"
}

ensure_repo() {
  if [ -d "$INSTALL_DIR/.git" ]; then
    log "updating existing repository at $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --tags --prune
    git -C "$INSTALL_DIR" checkout "$BRANCH"
    git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
  else
    log "cloning $REPO_URL into $INSTALL_DIR"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  fi
}

setup_env_file() {
  cd "$INSTALL_DIR"
  if [ ! -f .env.example ]; then
    log "missing .env.example template"
    return
  fi
  if [ -f .env ]; then
    log ".env already present"
    return
  fi
  log "creating .env from template"
  cp .env.example .env
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
from pathlib import Path
env_path = Path('.env')
text = env_path.read_text()
if 'REDIS_PASSWORD' in text:
    text = text.replace('REDIS_PASSWORD=""', 'REDIS_PASSWORD=""')
env_path.write_text(text)
PY
  fi
  log "update .env to match your environment if needed"
}

select_profile() {
  cd "$INSTALL_DIR"
  local summary="$1"
  if [ -n "$PROFILE_OVERRIDE" ]; then
    echo "$PROFILE_OVERRIDE"
    return
  fi
  local profile
  profile=$(python3 - "$summary" <<'PY'
import json
import sys
summary = json.loads(sys.argv[1])
gpu = summary.get('gpu', 'none')
if gpu == 'nvidia':
    print('gpu')
elif gpu == 'amd':
    print('rocm')
else:
    print('cpu')
PY
  )
  echo "$profile"
}

main() {
  ensure_repo
  cd "$INSTALL_DIR"

  if command -v sudo >/dev/null 2>&1 || [ "$(id -u)" -eq 0 ]; then
    log "ensuring operating system dependencies"
    bash ./tools/ensure_deps.sh || log "dependency installation requires manual intervention"
  else
    log "sudo not available; skipping dependency installation"
  fi

  HW_JSON=$(./tools/hw_probe.sh)
  log "hardware summary: $HW_JSON"
  PROFILE=$(select_profile "$HW_JSON")
  log "selected docker compose profile: $PROFILE"

  setup_env_file
  if [ -f .env ]; then
    python3 - "$PROFILE" <<'PY'
from pathlib import Path
import sys

profile = sys.argv[1]
env_path = Path('.env')
lines = env_path.read_text(encoding='utf-8').splitlines()
with env_path.open('w', encoding='utf-8') as handle:
    updated = False
    for line in lines:
        if line.startswith('PROFILE='):
            handle.write(f'PROFILE={profile}\n')
            updated = True
        else:
            handle.write(line + '\n')
    if not updated:
        handle.write(f'PROFILE={profile}\n')
PY
  fi
  mkdir -p sessions

  if [ "$PROFILE" = "gpu" ]; then
    if [[ "$(uname -s)" == "Linux" ]]; then
      if command -v sudo >/dev/null 2>&1; then
        log "ensuring NVIDIA container toolkit"
        sudo bash -c 'distribution=$(. /etc/os-release;echo $ID$VERSION_ID); curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg && curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && apt-get update && apt-get install -y nvidia-container-toolkit'
        sudo nvidia-ctk runtime configure --runtime=docker || true
        sudo systemctl restart docker || true
      else
        log "install NVIDIA container toolkit manually to enable GPU support"
      fi
    else
      log "GPU profile requested but host is not Linux; ensure Docker Desktop GPU support is enabled"
    fi
  fi

  log "building and starting containers"
  DOCKER_DEFAULT_PLATFORM=${DOCKER_DEFAULT_PLATFORM:-} docker compose --profile "$PROFILE" up -d --build

  cat <<INFO

HotMama is now running.
  API: http://localhost:8000/v1/healthz
  Queue: redis://localhost:6379

Next steps:
  * Monitor logs with: docker compose logs -f
  * Submit jobs via: curl -X POST http://localhost:8000/v1/jobs -d '{"payload":{"source_uri":"/path/to/video.mp4"}}' -H 'Content-Type: application/json'
  * Inspect artifacts in the ./sessions directory.
INFO
}

main "$@"
