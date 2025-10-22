# Migration Guide — Legacy HotMama → Robust Architecture

This guide highlights the major differences between the historical monolith and the new service-oriented stack.

## High-Level Changes

| Legacy | New |
| --- | --- |
| Single FastAPI/Gradio application (`app/`, `webapp/`) | Split into `deploy/api` (FastAPI gateway) and `deploy/worker-vision` (Redis worker) |
| In-process task execution | Async queue via Redis (`hotmama:jobs`) |
| Mixed dependencies & implicit installs | Deterministic `requirements.*.txt` pinned per service |
| Manual setup scripts | One-line bootstrap (`tools/bootstrap.sh`) with hardware detection |
| Scattered configs | Centralised `.env` with documented defaults |

## Launch Commands

### Old

```bash
python -m app.main
```

### New

```bash
# CPU (default)
docker compose up -d --build

# GPU (auto-selected by bootstrap or manual override)
docker compose --profile gpu up -d --build
```

The bootstrapper handles repository cloning, `.env` generation, dependency installation, and compose startup.

## Environment Variables

Legacy `.env` files contained numerous unused options. The new deployment focuses on:

- `PROFILE` — selects compose profile (`cpu`, `gpu`, `rocm`)
- `REDIS_URL`, `REDIS_QUEUE_NAME`, `REDIS_STATUS_PREFIX`
- `ARTIFACT_DIR`
- `LOG_JSON`

## Code Relocation

- Replace imports from `app.*` with the new API client or HTTP calls.
- Worker logic now resides under `deploy/worker-vision/app`. Custom processing hooks can be added within `processor.py`.
- Common helper scripts live in `deploy/scripts/` and `tools/`.

## Testing Changes

- Run unit tests with `pytest` (targets `tests/unit`).
- Integration coverage provided by `tools/test_compose_smoke.sh` (Docker required).
- Static analysis consolidated via `pyproject.toml` (ruff, black, mypy).

## Manual Migration Checklist

1. Install Docker Engine + Compose plugin (bootstrap script automates for Ubuntu/Debian).
2. Copy `.env.example` to `.env` and update overrides as needed.
3. Move any custom models/artifacts into `sessions/` or mount them via Compose overrides.
4. Update automation to call the REST API instead of invoking Python modules directly.

For additional help open an issue referencing `#robust-arch`.
