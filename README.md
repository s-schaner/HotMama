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

## Remote analysis workers (pull model)

Heavy inference never runs courtside. The host exposes a token-authed work
feed; any box the operator chooses (home GPU rig over ZeroTier, a cloud VM)
runs a worker that **pulls** rally chunks and posts observations back, which
land in the session's event log as `cv_observation` events with provenance
and confidence. This repo never configures or deploys to remote machines —
a worker is started by hand, where and when its operator decides.

```bash
# On the court host: enable the feed
HOTMAMA_WORKER_TOKEN=<shared-secret> hotmama serve

# On any analysis box (install: pip install "hotmama[worker,capture] @ git+...")
hotmama-worker --host http://<court-host>:8000 --token <shared-secret>
```

The default `stub` engine decodes each chunk and reports basic stats — it
proves the loop. Real CV engines plug in behind the same `AnalysisEngine`
protocol (`src/hotmama/worker/engine.py`) without touching the transport.

### The `vlm` engine (vision-language analysis)

Samples frames from each rally chunk (first and last always included) and
asks an OpenAI-compatible vision endpoint (vLLM) for a strict-JSON rally
summary — ball landed near/far, serve visible, jersey numbers seen. The
observations are deliberately attribution-free until court calibration
lands; team mapping is never guessed. Tiers live in a config file
(`examples/vlm-tiers.example.json` documents the current corona tiers):

```bash
# Sanity-check a tier's vision path (no court host needed)
hotmama-worker --probe-vlm --vlm-url http://corona:8005 --vlm-model qwen3-vl-8b

# Run a vision worker on the standard tier
hotmama-worker --host http://<court-host>:8000 --token <shared-secret> \
    --engine vlm --vlm-config examples/vlm-tiers.example.json --vlm-tier standard
```

The LLM set-summary feature can share the big tier — on the court host:
`HOTMAMA_LLM_PROVIDER=openai HOTMAMA_LLM_BASE_URL=http://corona:8000
HOTMAMA_LLM_MODEL=qwen3-vl-30b`.

### The `detect` engine (detection + tracking)

YOLO person detection (ultralytics, loaded lazily) fed through ByteTrack
(`trackers` package, with `supervision.Detections` as the currency). Emits
pixel-space `player_tracks` observations per rally chunk: persistent track
count, players visible avg/max, and a center-density grid — the seed of
position heatmaps. One-frame ghosts never count. Needs the `detect` extra
(`pip install "hotmama[detect,capture]"`, pulls torch):

```bash
hotmama-worker --host http://<court-host>:8000 --token <shared-secret> \
    --engine detect --detect-model yolo11n.pt --detect-stride 3
```

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
