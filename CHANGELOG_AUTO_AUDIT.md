# CHANGELOG_AUTO_AUDIT

## 2025-10-22
- fix: add context manager support to `SessionService` so web routes can use `with` safely.
- fix: repair analytics endpoints and static path handling in `webapp/server.py`, including absolute template/static paths and correct CSV export usage.
- chore: remove unused `frame_count` variable in `viz/trajectory.py` flagged by linting.
