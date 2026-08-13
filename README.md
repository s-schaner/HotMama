# HotMama v2 — Volleyball Analytics

Live volleyball statting and analytics that a coach can actually trust. Built for Mo.

- **Live statting** on a phone/tablet PWA — score, rotation, side-out %, moment tagging —
  every rally closeable in ~3 taps, every mistake correctable, nothing ever silently wrong.
- **Event-sourced engine**: every fact is an event in an append-only log; all stats are
  replayable projections. Corrections heal history.
- **Video from day one**: the session host records on a shared clock; tagged moments
  become clips.
- **Heavy AI off-box**: CV and VLM inference run on a home GPU rig over ZeroTier or on
  configurable cloud endpoints — the courtside laptop only captures, serves, and scores.

Full architecture and the decision log: [`docs/DESIGN.md`](docs/DESIGN.md).
Product requirements straight from the coach: [`docs/product/SPEC.md`](docs/product/SPEC.md).

## Layout

```
src/hotmama/core/       game engine — pure, event-sourced, exhaustively tested
src/hotmama/analytics/  projections: rotation table, player tallies, runs
src/hotmama/capture/    camera → segmented recording → event-driven clips
src/hotmama/server/     FastAPI + WebSocket + SQLite event store
ui/                     React + Vite PWA (coach view, statter view)
```

## Video capture

The host records the camera into rolling 60-second segments on its own clock —
the same clock that stamps every event — so clips are pure arithmetic:

- **Tag clips** (for humans): a `moment_tagged` event auto-cuts an H.264 clip
  around the moment; it appears in the coach UI seconds later, playable on any phone.
- **Rally chunks** (for the Well): every `rally_ended` cuts a stream-copy chunk
  of that rally — the payload the remote CV workers will consume.

Camera source is whatever the coach types: a USB device index (`0`), an
`rtsp://` stream, or a video file. ffmpeg comes bundled via `imageio-ffmpeg`
(the `capture` extra); a system ffmpeg is used when present.

## Development

Requires Python ≥ 3.11 and Node ≥ 20.

```bash
# Python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,capture]"
pytest
ruff check . && mypy

# Server (serves API + built UI)
hotmama serve  # or: uvicorn hotmama.server.app:create_app --factory --reload

# UI
cd ui && npm install && npm run dev
```

The v1 codebase (generic video-AI pipeline) was removed in the v2 rewrite; it remains in
git history at `c290465`.
