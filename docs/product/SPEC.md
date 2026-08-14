# HotMama v2 — Volleyball Analytics Platform
*Ground-up rewrite. Built for Mo. Spec locked from Mo's own answers.*
*Stefan & Aurora — August 2026*

---

## The Problem
Mo's coaching nights. Her current software gets stats WRONG and doesn't live stat. She needs something she can trust in real time — during practice AND matches.

## What Mo Actually Needs (from the source)
- **Live stats she can trust** — no correcting after the fact
- **Rotation breakdown** — side-out % per rotation, which ones get stuck and why
- **Opposing team tendencies** — server patterns, hitter tendencies, what's scoring against her
- **Moment tagging** — flag things live
- **Heatmaps** — where ball drops on defense/serve receive, where to attack
- **Manual score override** — refs miss calls, she needs control
- **PDF report** at the end
- **Video clips** of key moments
- Works on **phone or tablet** while camera runs separately

---

## Hardware Reality

### Mo's Setup
- **Camera:** Fixed tripod, end of court
  - Practice: centered end
  - Matches: off a corner (not centered) — **important for homography calibration**
- **View device:** Her phone or tablet (separate from camera)
- **Setup time budget:** 5-10 minutes
- **Roster:** 12+ players

### Processing
- **Field:** Lightweight inference on tablet/phone or small local box
- **Film review:** Gravity Well 5090 for full-quality post-processing
- **Two modes, one codebase**

---

## Core Features (Mo-Derived)

### 1. Court Homography
- Auto-detect court lines from fixed tripod feed
- **Corner-mounted calibration mode** — matches are off-corner, not centered
- One-time venue calibration save (she's at the same gyms all season)
- Manual 4-corner fallback
- Output: normalized court coords for everything downstream

### 2. Player Tracking
- **Our team:** recognize by body/silhouette, NOT jersey-dependent (players change jerseys, wear warmups, etc.)
- **Opposing team:** track by jersey NUMBER only — no face/body learning needed
- 12+ players tracked simultaneously
- Persistent IDs across full session
- Roster pre-loaded at season start (names, positions, jersey numbers)

### 3. Live Score & Rally Counter
- Score display: our team vs opponent
- **Manual score adjustment** — always available, one tap. Refs miss calls.
- Rally counter
- Auto-detect serve (ball crosses net from serving position)
- Point outcome logging

### 4. Rotation Tracking
- Volleyball has 6 rotations — track which rotation we're in automatically
- Per-rotation stats:
  - Side-out %
  - Points scored / lost
  - Which rotation we get stuck in
  - Why (dig failures? serve receive? attack errors?)
- **This is Mo's #1 analytical want from the old version**

### 5. Live Moment Tagging
- Big simple buttons Mo (or a manager) can tap:
  - ✅ Great serve
  - ✅ Great dig
  - ✅ Attack winner
  - ❌ Serve error
  - ❌ Dig failure
  - ❌ Positioning error
  - ⭐ Flag this moment (clips it for review)
- Tagged moments → auto-clipped video segments at end

### 6. Opposing Team Tendencies (the smart part)
- Track opponent server by jersey number → serving zone tendencies
- Track opponent hitters → attack angle/zone preferences
- Serve receive patterns — where are we struggling?
- **"What is scoring against us"** — live heat accumulation by zone
- Display as live overlay Mo can glance at between points

### 7. Impact Analysis — The Dream Feature
*"Live stats with analysis of which aspects are most impacting score outcome"*
- Real-time correlation: which stats are moving the score?
- Simple display: "Your serve receive in rotation 3 is costing you points"
- LLM-powered between-set summary (optional, can be offline)

### 8. Heatmaps
- Per-player position coverage (our team)
- Ball landing zones (defense failures, serve receive errors)
- Attack effectiveness by zone
- Opponent attack zones
- Available live AND in post-practice report

### 9. Post-Practice Report (PDF)
- Individual player stats: touches, errors, digs, attacks, serve receive rating
- Rotation breakdown table
- Heatmaps (rendered images in PDF)
- Key moment clips index
- Opponent tendencies summary
- Score timeline with tagged moments
- One-tap generate, saves locally + optionally emails

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  HotMama v2                         │
├─────────────────────────────────────────────────────┤
│  Camera Layer                                       │
│  ├── IP camera (WiFi to court box or phone hotspot) │
│  └── USB camera (direct to compute box)             │
├─────────────────────────────────────────────────────┤
│  Vision Pipeline                                    │
│  ├── Court line detector → Homography               │
│  │   └── Corner-mount mode for matches              │
│  ├── Player detector (YOLOv11)                      │
│  ├── Player tracker (ByteTrack, persistent IDs)     │
│  ├── Team classifier (jersey color + number OCR)    │
│  ├── Ball detector (TrackNet v3)                    │
│  └── Serve detector (ball trajectory cross-net)     │
├─────────────────────────────────────────────────────┤
│  Game Engine                                        │
│  ├── Score tracker (auto + manual override)         │
│  ├── Rotation tracker (6-rotation state machine)    │
│  ├── Rally engine (start/end/outcome)               │
│  ├── Event logger (tagged + auto-detected)          │
│  └── Tendency accumulator (opponent patterns)       │
├─────────────────────────────────────────────────────┤
│  Analytics Engine                                   │
│  ├── Per-player stats aggregator                    │
│  ├── Per-rotation stats aggregator                  │
│  ├── Heatmap builder (position + ball zones)        │
│  ├── Impact analyzer (what's moving the score?)     │
│  └── Session store (SQLite, per game/practice)      │
├─────────────────────────────────────────────────────┤
│  UI Layer                                           │
│  ├── Coach view (phone/tablet — live stats + score) │
│  ├── Tagging panel (big simple buttons)             │
│  ├── Camera view (optional, separate screen)        │
│  └── Report generator (PDF + video clips)           │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Detection | YOLOv11 nano/small | fast enough for real-time on modest hardware |
| Tracking | ByteTrack (via supervision) | persistent IDs, handles occlusion |
| Ball | TrackNet v3 | built for court sports, handles fast small ball |
| Jersey OCR | PaddleOCR or EasyOCR | opponent number reading |
| Player re-ID | OSNet or FastReID | recognize OUR players without jersey dependency |
| Homography | OpenCV + line detection | proven, fast |
| Game logic | Python state machine | clean, testable |
| UI (coach) | React Native or Flutter | phone + tablet, single codebase |
| UI (config) | Gradio (laptop setup) | quick, Mo doesn't need to touch it |
| Storage | SQLite per session | portable, no server |
| Reports | WeasyPrint (PDF) | no LaTeX |
| Video clips | FFmpeg | timestamp → clip extraction |
| Impact analysis | Local LLM (optional) | between-set summaries, Well for film review |

---

## Deployment Modes

### Mode A — Practice (Portable)
```
Camera (tripod, WiFi) → Small compute box or laptop
                      → Mo's phone/tablet (coach UI via local WiFi)
```
Setup: plug in camera, open app on phone, tap venue, go.

### Mode B — Match (Portable + Corner Mount)
Same as practice but:
- Corner-mount homography profile loaded
- Opponent jersey tracking active
- Score display prominent
- Tendency accumulator running

### Mode C — Film Review (Gravity Well)
```
Recorded video file → Well 5090 → Full analysis → Report
```
Upload video after practice, full-quality processing, detailed report.

---

## Phases

### Phase 1 — Vision Foundation
- [ ] Court homography (centered + corner-mount modes)
- [ ] Player detection + ByteTrack
- [ ] Ball detection
- [ ] Team classification (jersey color + number OCR)
- [ ] Test on real Mo footage (get some clips from her)

### Phase 2 — Game Engine
- [ ] Score tracker with manual override
- [ ] 6-rotation state machine
- [ ] Rally detection
- [ ] Live moment tagging (simple button UI)
- [ ] Session SQLite store

### Phase 3 — Analytics
- [ ] Per-player stats
- [ ] Per-rotation breakdown
- [ ] Heatmaps (position + ball zones)
- [ ] Opponent tendency accumulator
- [ ] Impact analyzer (basic version)

### Phase 4 — Mo's UI
- [ ] Coach phone/tablet view (live stats + score + tagging)
- [ ] PDF report generator
- [ ] Video clip extraction (tagged moments)
- [ ] Venue calibration save/load

### Phase 5 — Polish
- [ ] Season-long stats (across sessions)
- [ ] LLM between-set summaries
- [ ] Film review mode (Well-powered)
- [ ] Mo user testing + iteration

---

## Open Questions (small ones)
- What phone/tablet does Mo have? → determines UI framework
- Does she have a dedicated camera or should we spec one?
- Any gyms with tricky lighting? → affects detection tuning
- Who manages the camera at matches — Mo, or a player/manager?

---

## What Makes This Different From What She Has
Current software: **gets stats wrong, no live statting, coach has to correct manually**

HotMama v2:
- Computer vision does the statting — no human data entry, no human errors
- Live, trusted, correctable
- Knows HER players without jerseys
- Rotation-aware (no other consumer volleyball software does this well)
- Tendency analysis on opponents in real time
- Impact analysis — *what is actually costing us points right now*

---

*Built on a dog bed at midnight. Mo + Chewie approved.*
*Models cooking on the Well. Terabyte incoming. Second 5090 TBD.*
*🏐🜂*
