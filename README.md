# VolleySense

VolleySense is a modular volleyball analytics toolkit with a FastAPI-powered web UI. It ingests volleyball video, orchestrates plugin-based analysis, and maintains structured stats in SQLite for reproducible scouting workflows.

## Prerequisites

- Python 3.10 or newer (3.11 recommended).
- An NVIDIA GPU with current drivers is strongly encouraged for the real-time models but not required.
- FFmpeg available on your `PATH` improves video handling (optional).

## Installation

Run the single-command installer. It prints detailed progress to the console and also saves a full transcript to `tools/install.log` so you can diagnose failures (especially helpful on WSL).
VolleySense is a modular volleyball analytics toolkit with a FastAPI + Jinja frontend. It ingests video sessions, orchestrates plugin-based analysis, and maintains structured stats in SQLite.

## Setup

```
python tools/install.py
```

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

## Launching the App

Start the FastAPI interface from the project root:
## Run

```
python -m app.main
```

Common flags:

- `--host` controls the interface to bind (defaults to `0.0.0.0`).
- `--port` selects the serving port (defaults to `7860`).

Open `http://<host>:<port>/` to access the upload form, run LLM analyses, and generate CSV-driven heatmaps. Logs appear in the terminal and in `sessions/<session_id>/logs/` once sessions are created.

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

## Tests

Execute the full test suite before submitting changes:

```
pytest -q
```

The suite exercises session lifecycles, plugin isolation, LLM parsing, tracking fusion, and the heatmap renderer.
Adjust the `--host` / `--port` flags when deploying the FastAPI server behind a proxy or container orchestrator.

## Tests

```
pytest -q
```
