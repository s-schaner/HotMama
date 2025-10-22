# CHANGELOG_AUTO_AUDIT

## 2025-10-22
- fix: add context manager support to `SessionService` so web routes can use `with` safely.
- fix: repair analytics endpoints and static path handling in `webapp/server.py`, including absolute template/static paths and correct CSV export usage.
- chore: remove unused `frame_count` variable in `viz/trajectory.py` flagged by linting.
- fix: restore missing stdlib imports in `core/analysis.py` to prevent runtime NameErrors when rendering heatmaps.
- fix: harden plugin API models by importing `Path` and using `Field(default_factory=...)` for mutable collections.
- fix: patch session API typing/logging regressions so FastAPI routers serialize `SessionDetailsResponse` correctly.
- fix: await `validate_video_upload` results and persist the validated bytes so uploaded videos reach the ingest pipeline.
- chore: ignore local `.venv/` environments in git metadata.
