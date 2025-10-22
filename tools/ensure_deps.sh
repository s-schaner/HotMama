#!/usr/bin/env bash
# Installs prerequisite tooling for HotMama deployment.

set -euo pipefail

OS=""
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS=$ID
fi

SUDO=sudo
if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
fi

echo "[ensure_deps] ensuring base packages for $OS"

if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y ca-certificates curl gnupg lsb-release
  if ! command -v docker >/dev/null 2>&1; then
    ${SUDO} install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/$OS/gpg | ${SUDO} gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS $(lsb_release -cs) stable" | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
elif [[ "$OS" == "fedora" || "$OS" == "centos" ]]; then
  echo "[ensure_deps] please install Docker manually for $OS" >&2
else
  echo "[ensure_deps] unsupported distribution: $OS" >&2
fi

if command -v wsl.exe >/dev/null 2>&1; then
  echo "[ensure_deps] detected WSL; ensure \"wsl --install --web-download\" has completed."
fi
