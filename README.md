# HotMama — Robust Vision Pipeline

> Deterministic, hardware-aware deployment for the HotMama computer-vision stack.

## 🚀 One-Line Install

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/s-schaner/HotMama/main/tools/bootstrap.sh)"
```

The installer clones the repository, probes hardware, installs Docker prerequisites, selects the correct compose profile (CPU/NVIDIA GPU/experimental ROCm), and launches the stack.

## 🧱 Architecture Overview

```
deploy/
  api/           FastAPI gateway (Python 3.11, CPU)
  worker-vision/ Vision workers (CPU + CUDA targets)
  gui/           Gradio control surface (Python 3.11, CPU)
  scripts/       Shared runtime helpers (Redis wait, etc.)
  base-images/   Documentation for base container pins
```

Supporting infrastructure:

- **Redis 7.2.4 (alpine)** provides the job queue and status storage.
- **docker-compose** orchestrates services with profiles:
  - `default` → API + Redis + CPU worker
  - `gpu` → API + Redis + CUDA worker (auto-selected when NVIDIA GPUs detected)
  - `rocm` → reserved for future AMD builds (marked experimental)
- Artifacts are persisted under `./sessions/` and shared between services.
- Optional **GUI**: visit `http://localhost:7860` (default) to queue jobs and inspect artifacts.

## 🔁 Runtime Flow

1. API receives a job (`POST /v1/jobs`) and enqueues a JSON payload in Redis.
2. Worker pops jobs, lazily loads the Torch/ONNX/OpenCV toolchain, processes the input, and writes a result artifact (`result.json`).
3. API exposes job status (`GET /v1/jobs/{id}`) and serves artifacts once complete.

Both services share configuration through `.env` (generated automatically). Logging can be toggled to JSON by setting `LOG_JSON=1`.

## 🛠 Development

Clone manually if preferred:

```bash
git clone https://github.com/s-schaner/HotMama.git
cd HotMama
cp .env.example .env
```

Install Python tooling (requires 3.11):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deploy/api/requirements.api.txt \
            -r deploy/worker-vision/requirements.worker.cpu.txt
# Optional CUDA tooling for local GPU runs
# pip install -r deploy/worker-vision/requirements.worker.cuda.txt
pip install fastapi[all] pytest ruff black mypy  # development extras
```

> The production containers remain slim; install only the dev dependencies you need locally.

### Local Docker Compose

Launch helpers are provided under `tools/` so you can spin up the full stack with a single command:

```bash
./tools/launch_cpu.sh
```

The CPU script builds the images, exports `PROFILE=cpu`, and brings up the API, Redis, worker, and GUI services. To exercise the CUDA worker, use the GPU variant (requires NVIDIA Container Toolkit):

```bash
./tools/launch_gpu.sh
```

Both scripts print follow-up log commands once the containers are running. You can still call `docker compose` directly if you prefer manual control.

### Testing & Quality Gates

```bash
ruff check .
black --check .
mypy deploy/api deploy/worker-vision
pytest
```

Run the integration smoke test (uses Docker):

```bash
./tools/test_compose_smoke.sh
```

### Elegant Shutdown

Allow in-flight jobs to finish, then stop the stack cleanly:

```bash
docker compose stop
docker compose down --remove-orphans
```

`docker compose stop` sends a `SIGTERM` to each service so workers can flush logs and close Redis connections. `docker compose down --remove-orphans` then tears down the containers and network once everything has exited. If you launched with the GPU profile, repeat the commands with `--profile gpu` to ensure the CUDA worker shuts down gracefully.

## 📡 API Quick Reference

- `GET /v1/healthz` — health check
- `POST /v1/jobs` — enqueue job (`{"payload": {"source_uri": "/path/video.mp4"}}`)
- `GET /v1/jobs/{id}` — job status
- `GET /v1/jobs/{id}/artifact` — download artifact

## 🔐 Configuration Matrix

| Variable | Default | Description |
| --- | --- | --- |
| `PROFILE` | `cpu` | docker-compose profile (`cpu`, `gpu`, `rocm`) |
| `REDIS_URL` | `redis://queue:6379/0` | Redis connection string |
| `REDIS_QUEUE_NAME` | `hotmama:jobs` | Redis list for job dispatch |
| `REDIS_STATUS_PREFIX` | `hotmama:job` | Prefix for job metadata hashes |
| `ARTIFACT_DIR` | `/app/sessions` | Shared artifact directory |
| `LOG_JSON` | `0` | Emit JSON logs when set to `1` |
| `GUI_API_BASE_URL` | `http://api:8000/v1` | GUI → API endpoint |
| `GUI_PORT` | `7860` | GUI listening port |

## 🧭 Troubleshooting

- **Docker not found** — rerun the bootstrap script; it installs Docker CE on Ubuntu/Debian and prints instructions for other OSes.
- **GPU not used** — ensure `nvidia-smi` works on the host and rerun bootstrap. The script installs NVIDIA Container Toolkit on Linux (when sudo/root available).
- **Artifacts missing** — check worker logs (`docker compose logs worker-vision-cpu`). The worker stores artifacts under `./sessions/<job-id>/result.json`.
- **WSL2 GPU** — enable GPU support in Windows settings and ensure the NVIDIA drivers are installed in Windows.

## 📄 Additional Docs

- [`MIGRATION.md`](MIGRATION.md) — map from the legacy monolith to the new services.
- [`CHANGELOG_AUTO_AUDIT.md`](CHANGELOG_AUTO_AUDIT.md) — structured change log for automated audits.
- [`AUTO_AUDIT_REPORT.md`](AUTO_AUDIT_REPORT.md) — human-readable summary with risk notes.

## 📜 License

This project retains the original licensing terms of HotMama. Refer to the repository history for legacy components.
