#!/usr/bin/env sh
# Hardware probing utility emitting JSON summary.

set -eu

json_escape() {
  printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

OS_NAME="unknown"
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [ "${PRETTY_NAME:-}" != "" ]; then
    OS_NAME=$PRETTY_NAME
  else
    OS_NAME=${NAME:-unknown}-${VERSION_ID:-}
  fi
else
  OS_NAME=$(uname -s)
fi

GPU="none"
CUDA=""
DRIVER=""
ROCM=""

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="nvidia"
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 2>/dev/null || printf '')
  CUDA=$(nvidia-smi | awk '/CUDA Version/ {print $3; exit}' 2>/dev/null || printf '')
fi

if [ "$GPU" = "none" ] && command -v rocminfo >/dev/null 2>&1; then
  GPU="amd"
  ROCM=$(rocminfo 2>/dev/null | awk '/ROCm version:/ {print $3; exit}' || printf '')
fi

CONTAINER_RUNTIME="none"
if command -v docker >/dev/null 2>&1; then
  CONTAINER_RUNTIME="docker"
fi

printf '{"gpu":%s,"cuda":%s,"driver":%s,"rocm":%s,"os":%s,"container_runtime":%s}\n' \
  "$(json_escape "$GPU")" \
  "$( [ -n "$CUDA" ] && json_escape "$CUDA" || printf 'null')" \
  "$( [ -n "$DRIVER" ] && json_escape "$DRIVER" || printf 'null')" \
  "$( [ -n "$ROCM" ] && json_escape "$ROCM" || printf 'null')" \
  "$(json_escape "$OS_NAME")" \
  "$(json_escape "$CONTAINER_RUNTIME")"
exit 0
