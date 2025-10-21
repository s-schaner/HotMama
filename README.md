# VolleySense

VolleySense is a modular volleyball analytics toolkit with a FastAPI-powered web UI. It ingests volleyball video, orchestrates plugin-based analysis, and maintains structured stats in SQLite for reproducible scouting workflows.

## Prerequisites

- Python 3.10 or newer (3.11 recommended).
- An NVIDIA GPU with current drivers is strongly encouraged for the real-time models but not required.
- FFmpeg available on your `PATH` improves video handling (optional).

## Installation

Run the single-command installer. It now performs a lightweight GPU driver scan, locks dependency versions to tested combinations, and bails out early when the active Python version is outside the supported window. Detailed progress is echoed to the console and mirrored to `tools/install.log` so you can diagnose failures (especially helpful on WSL).
VolleySense is a modular volleyball analytics toolkit with a FastAPI + Jinja frontend. It ingests video sessions, orchestrates plugin-based analysis, and maintains structured stats in SQLite.

## Setup

```
python tools/install.py
```

Pass `--profile cpu|nvidia|amd` to override the auto-detected dependency profile, or `--attempt-driver-adjust` if you would like the installer to poke the available GPU management tools (for example, enabling NVIDIA persistence mode when supported). These switches are optional; by default the script records diagnostics without touching the driver stack.

The installer attempts to:

1. Install core Python dependencies such as FastAPI, Uvicorn, Ultralytics, and NumPy.
2. Resolve OpenCV with a headless build first, then fall back to the desktop build if necessary.
3. Pull GPU-accelerated extras (PyTorch CUDA wheels, ONNX Runtime GPU, TensorRT, FlashAttention) on a best-effort basis.
4. Verify critical imports (`fastapi`, `numpy`, `matplotlib`).

If a package fails, consult the tail of the stdout/stderr block in the terminal or open `tools/install.log` for the complete history. The script exits non-zero when a core dependency cannot be installed or imported.

FlashAttention is optional and will be skipped on unsupported combos (for example, early Blackwell `sm_120`). If you need it, make sure you're on a supported CUDA/PyTorch pair and that a prebuilt wheel exists for your architecture; otherwise a source build may require matching CUDA toolkits and compilers.

### Windows Subsystem for Linux (WSL)

When running inside WSL, ensure that:

- You have activated your Python environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
- `pip` is up to date (`python -m pip install --upgrade pip`).
- GPU access is configured via the NVIDIA WSL drivers if you expect CUDA acceleration.

The installer surfaces the detected Python executable and platform in the log header so you can confirm WSL-specific paths.

## Configuration

VolleySense supports environment-based configuration for easy customization and deployment.

### Environment Variables

Copy `.env.example` to `.env` and customize settings:

```bash
cp .env.example .env
```

Key configuration options:

- **Server**: `VOLLEYSENSE_HOST`, `VOLLEYSENSE_PORT`, `VOLLEYSENSE_DEBUG`
- **Upload Limits**: `VOLLEYSENSE_MAX_FILE_SIZE_MB` (default: 500MB)
- **Rate Limiting**: `VOLLEYSENSE_RATE_LIMIT_PER_MINUTE` (default: 10 requests/min)
- **LLM Timeout**: `VOLLEYSENSE_LLM_TIMEOUT_SECONDS` (default: 120s)
- **CORS**: Set `VOLLEYSENSE_ENABLE_CORS=true` for cross-origin requests

See `.env.example` for all available options.

## Security Features

VolleySense includes enterprise-ready security features:

- **Input Validation**: File type and size validation for uploads
- **Rate Limiting**: API rate limiting to prevent abuse (configurable per-IP)
- **Secure File Handling**: Cryptographically secure temporary file names
- **Structured Error Handling**: Consistent error responses with appropriate HTTP status codes
- **CORS Support**: Optional CORS configuration for web integrations

## Launching the App

Start the FastAPI interface from the project root:

```
python -m app.main
```

Common flags:

- `--host` controls the interface to bind (defaults to `0.0.0.0`).
- `--port` selects the serving port (defaults to `7860`).

Open `http://<host>:<port>/` to access the upload form, run LLM analyses, and generate CSV-driven heatmaps. The Analyze form now includes a preset selector that remembers common endpoint/token/model combinations (see `sessions/endpoint_presets.json`), so you can toggle between the default Hugging Face endpoint and a local LM Studio server without retyping credentials. Logs appear in the terminal and in `sessions/<session_id>/logs/` once sessions are created.

## Docker usage

The project ships with a containerized runtime. Build the image from the repository root:

```
docker build -t volleysense .
```

Run it with your preferred port mapping (mount the `sessions/` directory if you want to persist artifacts on the host):

```
docker run --rm -p 8000:8000 -v "$(pwd)/sessions:/app/sessions" volleysense
```

On startup the container logs a hardware summary, the active dependency profile, and the URL for the GUI (`http://localhost:8000/`). Detailed runtime logs live under `logs/container.log`, while crash diagnostics are mirrored to `logs/container-crash.log` if the server exits unexpectedly.

## Typical Workflow

1. **Create or load a session** in the Session Manager tab.
2. **Assign teams and rosters** on the Roster tab (CSV import/export coming soon).
3. **Upload clips** through the Clip Manager; metadata is stored in SQLite and artifacts land in `sessions/<session_id>/clips/`.
4. **Analyze rallies** in the Analysis panel by selecting a clip, configuring the LLM backend (HF endpoint or local OpenAI-compatible server), and prompting for rally summaries. Validated outputs update the stats rollup automatically.
5. **Inspect visuals** using the Heatmap and overlay tabs. Heatmap CSV uploads should contain `x,y,h` columns in meters; generated PNGs are saved alongside the session artifacts.
6. **Extend functionality** by dropping additional plugins into `plugins/<name>/plugin.py`. They are auto-discovered and isolated so a failing plugin cannot crash the UI.

## Troubleshooting

- Review `tools/install.log` for installation diagnostics.
- Run `python tools/diagnose.py` to collect environment info (CUDA, drivers, FFmpeg).
- Ensure SQLite writes are permitted; the app stores state under `sessions/`.
- For GPU issues, verify CUDA availability via `python -c "import torch; print(torch.cuda.is_available())"`.

## API Endpoints

### Core Endpoints

- `GET /` - Main web interface
- `GET /health` - Health check endpoint (returns JSON status)
- `GET /docs` - OpenAPI documentation (when debug mode enabled)
- `POST /_api/analyze` - Video analysis endpoint
- `POST /_api/heatmap_pipeline` - Ball trajectory detection and heatmap generation

### Health Check

The `/health` endpoint provides application status for monitoring and container orchestration:

```bash
curl http://localhost:7860/health
```

Returns:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "settings": {
    "sessions_dir": "/app/sessions",
    "max_file_size_mb": 500,
    "rate_limiting_enabled": true
  }
}
```

## Development

### Running Tests

Execute the full test suite before submitting changes:

```bash
pytest -q
```

Run tests with coverage:

```bash
pytest --cov=. --cov-report=html
```

The suite exercises session lifecycles, plugin isolation, LLM parsing, tracking fusion, and the heatmap renderer.

### CI/CD

The project includes a GitHub Actions CI pipeline (`.github/workflows/ci.yml`) that runs:

- **Linting**: ruff, black, mypy
- **Testing**: pytest across Python 3.10 and 3.11
- **Docker**: Build verification
- **Security**: bandit security scanning

The pipeline runs on all pushes to `main`, `develop`, and `claude/**` branches.
