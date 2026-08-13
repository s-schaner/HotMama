# HotMama v2 — Architecture & Decisions

*Ground-up rewrite of HotMama as a volleyball analytics platform. Product requirements
live in [`docs/product/SPEC.md`](product/SPEC.md) (derived from Mo's questionnaire answers,
preserved in [`docs/product/MO_RESPONSES.md`](product/MO_RESPONSES.md)).*

This document records the architecture and the decisions that shape it. When a decision
here conflicts with the SPEC, this document wins — each deviation is called out and justified.

---

## 1. The governing principle: the event log is the spine

Mo's core complaint about her current software: **it gets stats wrong and she can't
live-stat**. Trust is the product. The architecture that earns trust:

- Every fact in a session is an **immutable event** in an append-only log:
  `rally_ended`, `moment_tagged`, `score_adjusted`, `sub_made`, …
- Producers of events are interchangeable: a coach's thumb, a statter's tablet,
  a CV pipeline on the Well, a film-review import. The engine does not care.
- **All derived state is a pure function of the log.** Score, rotation, side-out %,
  heatmaps, reports — everything is recomputable by replay.
- **Corrections are events too** (`event_retracted`, `score_adjusted`). Fixing a
  mistake retroactively heals every downstream stat. Correction isn't a patch on the
  system; it *is* the system.
- Future CV events carry `producer` + `confidence` metadata. Low-confidence events are
  surfaced as "confirm?" prompts, never silently committed. The system is
  **manual-first with CV assist**, not CV-first with manual cleanup.

Consequences we get for free:
- **Rotation analytics — Mo's #1 want — requires zero computer vision.** Rotation
  advances deterministically on side-outs; given a starting lineup, the state machine
  derives everything.
- Film-review mode is the same engine fed by a different producer at a different speed.
- Multi-device (coach + statter simultaneously) is just multiple writers to one log.

## 2. System shape

```
                     gym (LAN / hotspot)                      house (over ZeroTier)
┌─────────────────────────────────────────────────────┐   ┌──────────────────────────┐
│  Laptop = session host                              │   │  The Well (RTX 5090)     │
│  ├── hotmama.server  FastAPI + WebSocket + SQLite   │◄──┼── CV workers: detection, │
│  ├── hotmama.core    game engine (pure, replayable) │   │   tracking, OCR, re-ID   │
│  ├── hotmama.capture camera → rally-aligned clips   │───┼─► rally clips in,        │
│  └── serves the PWA to any device on the LAN        │   │   gated events out       │
│                                                     │   └──────────────────────────┘
│  Phones/tablets (PWA over LAN):                     │   Optional cloud VLM/LLM:
│  ├── Coach view    score, dashboards, tags, undo    │   OpenAI/Anthropic/xAI/Gemini
│  └── Statter view  rally entry flow                 │   behind the same provider API
└─────────────────────────────────────────────────────┘
```

- The **laptop hosts everything** at the gym. A mini-PC appliance can replace it later —
  identical stack.
- **Heavy inference never runs on the laptop.** CV/VLM work goes to the Well over
  ZeroTier, or to a configurable cloud endpoint. Latency is acceptable; a laptop GPU
  hosting the big models is not realistic.
- The UI is a **web PWA served by the host** — zero install, works on whatever device
  Mo has, one codebase for coach view, statter view, and setup/calibration screens.
  *(Deviation from SPEC's "React Native or Flutter" — approved 2026-08-13.)*

## 3. Decision log

| # | Decision | Choice | Why / notes |
|---|----------|--------|-------------|
| D1 | Build order | **Engine-first**, vision lands on the proven spine | Deviation from SPEC phases (approved). No Mo footage in hand yet; engine-first ships a usable, never-wrong live-statting tool early; rotation analytics needs no CV. |
| D2 | Coach UI | **Web PWA** (React + Vite, served from host) | Deviation from SPEC (approved). Kills the "what device does Mo have" question; no app store; setup screens share the codebase. |
| D3 | Session host | **Laptop** running the Python stack | Approved. Phone-only/serverless mode rejected (would force the engine into TS and split the source of truth). |
| D4 | Heavy compute | **Remote inference**: the Well over ZeroTier, plus pluggable cloud endpoints (OpenAI/Anthropic/xAI/Gemini) behind one provider interface | Approved. Laptop does capture + engine + serving only. |
| D5 | Statting roles | **Coach view and statter view both exist day one**; multi-writer event log | Approved. |
| D6 | Video | **Recorded from day one** by the host on a shared clock; tagged moments auto-clip | Approved. |
| D7 | Posture | **Product-shaped**: real entities (org/team/season/opponent/user), global IDs, tenancy-ready schema | Approved. Mo is user #1, not the only user ever. |
| D8 | Live CV transport | **Rally-chunked near-live** (default): rally ends → clip ships to the Well → gated events merge back seconds later. Continuous streaming rejected as fragile; full film-review remains available | Default chosen by Claude, open to veto. Degrades gracefully: bad uplink = analysis arrives later; live statting never blocks. |
| D9 | Camera source | **Abstracted capture source** (USB index / RTSP URL / file import); USB rig assumed for day one | Default chosen by Claude, open to veto. |
| D10 | Rally entry depth | **Progressive**: every rally closeable in ~3 taps (outcome → reason → player), expandable touch chain when the statter has bandwidth | Default chosen by Claude, open to veto. Data quality scales with the human; UI never forces depth. |
| D11 | Auth day one | **Local-first, sync-ready**: users/roles (coach/statter/viewer) in schema, lightweight PIN login on the host, no cloud ops before first practice | Default chosen by Claude, open to veto. |
| D12 | Storage | **SQLite** (WAL) on the host; one DB, tenancy-ready; sessions exportable | Per SPEC. |
| D13 | Player identity | Our team via body-based re-ID (never faces), opponents by jersey number only; all processing local/on-Well | Per SPEC, with the explicit "no face recognition" guardrail. |
| D14 | Impact analysis v1 | **Descriptive attribution** ("4 of 6 points lost in rotation 3 were serve-receive"), not statistical correlation — a set is ~25 points; correlation on that sample is noise. LLM narration optional on top | Refinement of SPEC's dream feature. |

## 4. Domain model

Entities (product-shaped from day one, all with global IDs):

- **Org → Team → Season → Player** (jersey, positions, libero flag)
- **Opponent** (per season; opponent players identified by jersey number only)
- **Session** = one match or practice. `session_kind: match | practice | scrimmage`
- **Event** = the log. `(session_id, seq, event_id, type, payload, occurred_at,
  recorded_at, producer, actor, confidence, retracted_by)`
- **MatchState** = derived, never stored authoritatively: score, current set,
  serving team, our rotation (1–6), lineup on court, per-rotation tallies, tag index.

### Volleyball rules encoded in the engine
- Rally scoring; sets to 25 (set 5 to 15), win by 2; best-of-5 or best-of-3 configurable;
  practices can run unscored segments or games to N.
- Rotation advances **only on side-out** (winning a rally while receiving); serving team
  holds rotation while scoring.
- Lineups are per-set; subs swap on-court identity; libero handling is modeled
  (replacement doesn't consume a sub, libero can't rotate to front row).
- Manual score adjustment is always legal and is its own event type.

### Event types (v1)
`session_created`, `roster_registered`, `set_started` (lineup, first server),
`rally_started`, `rally_ended` (winner, reason, key player, optional touch chain),
`score_adjusted`, `moment_tagged`, `sub_made`, `libero_swap`, `timeout_called`,
`set_ended`, `session_closed`, `event_retracted`, `note_added`.

Reserved for the CV phase: `cv_observation` (producer=well|cloud, confidence, kind:
serve_detected / touch_detected / ball_landed / player_position, …) — these become
*proposals* the UI confirms into domain events when confidence is below the auto-commit
threshold.

## 5. Repository layout

```
src/hotmama/
  core/        # pure domain: events, reducer, rules, rotation — zero I/O deps, mypy strict
  analytics/   # projections over the log: rotation table, player tallies, runs, impact v1
  server/      # FastAPI app, REST + WebSocket, SQLite store, settings
  capture/     # (next) camera sources, rally-aligned recording, clip extraction
  inference/   # (next) provider-abstracted VLM/LLM + Well worker protocol
ui/            # React + Vite + TS PWA (coach, statter, setup)
docs/          # this file + product docs
tests/         # pytest: core is exhaustively tested; server via TestClient
```

## 6. Delivery phases (engine-first)

1. **Spine (this phase)** — core engine + SQLite event store + FastAPI/WS +
   coach & statter PWA views. Usable at a real practice with manual statting.
2. **Video** — capture module records on the host clock; tagged moments auto-clip
   (FFmpeg); rally-aligned chunking ready for the Well.
3. **Analytics & report** — rotation table, player stats, PDF report (WeasyPrint),
   clip index, impact v1.
4. **Well integration** — ZeroTier worker protocol, court homography + calibration UI,
   detection/tracking/OCR/re-ID rolling in one detector at a time behind confidence gates.
5. **Polish** — season aggregation, LLM between-set summaries, film-review mode,
   opponent tendency dashboards.

## 7. Salvage from v1 (patterns, not code)

Catalogued before demolition (git history retains everything at `c290465`):
- Frame sampling + SSE streaming VLM client (`deploy/gui/app/client.py:504–601`) —
  basis for `inference/` providers. Fix: sample the final frame, preserve aspect ratio,
  honor the frame-count parameter.
- NL → schema-validated JSON with `json_schema` response_format + pydantic revalidation
  (`deploy/api/app/parsing.py:89–125`) — reuse for LLM-composed configs/summaries.
  Fix: real strict-mode schemas (`additionalProperties: false`, all-required).
- Two-pass cheap-model-then-big-model enrichment (`client.py:178–195`).
- `httpx.MockTransport` test idiom and stateful fakes (`tests/unit/gui/test_client.py:18–65`).
- Codec-fallback VideoWriter ladder (`processor.py:270–308`) for the capture module.
- `hw_probe.sh` hardware detection JSON (`tools/hw_probe.sh`).
- Config idioms: `AliasChoices` migration aliasing, `extra="forbid"` on LLM-facing
  models, `Field(alias="ENV_NAME")` settings.
- Known cruft that must not return: hardcoded LAN IPs, emoji in return values,
  `datetime.utcnow()`, fake overlays/stub models.
