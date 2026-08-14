# Real-Footage Validation — 2026-08-14

Four clips from Stefan (repo root on `main`): three 1080p30 from a tournament
venue (elevated, roughly centered end-court, wide-angle lens, wood court on
blue surround) and one school-gym screen recording (off-corner elevated view,
glossy floor with glare, painted lines). Together they cover both SPEC camera
scenarios. All numbers below: yolo11n unless stated, CPU inference.

## What works

- **Calibration (near-half mode) is solid.** Four eyeballed corners on the
  tournament clip put every projected court line on the real paint/boundary.
  The mapping is metrically sane: net at 9.00m, far extrapolation smooth.
- **Detection at the right imgsz is complete for the near half.** At
  `imgsz=1280-1600`, every on-court near-half player is boxed with 0.4-0.9
  confidence. At the ultralytics default 640 the same frames lose most
  players — the engine default is now 1280.
- **Spectator filtering falls out of calibration.** A 26s tournament rally
  clip produced 862 in-court foot hits and 1,406 out-of-bounds hits
  (sideline spectators, refs, the near-camera person) — cleanly excluded by
  court bounds, no special-casing.
- **End-to-end cost is fine.** 26s clip, stride 5, imgsz 1280: ~40s on CPU
  (~240ms/frame). Any GPU worker does this near-live with margin.

## Hard limits found (and what moves them)

- **The far back row is below the detection floor** on this footage: zero
  detections beyond the net region even with yolo11s at imgsz 1920,
  conf 0.15. Low mount + wide angle compress the far half into ~60px of
  1080p; far back-row players are ~25-35px and part-occluded by the net and
  front row. A human squints at the same pixels. Axes that move this:
  camera height (biggest lever), longer focal length / higher resolution
  (the planned "way nicer camera"), larger models on GPU workers, and VLM
  narration for far-half context.
- **Wide-angle barrel distortion** bows the near baseline slightly under a
  straight-line homography. Mid-court accuracy is good; frame-edge accuracy
  degrades. Lens undistortion remains a planned refinement.

## Bugs caught by validation

- `CourtCalibration.scaled_to()` dropped `mode`, silently remapping
  near-half calibrations as full-court downstream. Fixed + regression test.
- Tracker activation at 0.4 erased legitimate 0.25-0.4-confidence
  detections; engine now activates at 0.25 and relies on persistence
  filtering for noise.

## Corner/side mounts (school-gym clip)

The school-gym clip is an elevated sideline-corner view: the net crosses
mid-frame and **one court corner is cut off outside the frame**. Detection
handles the glossy floor and glare fine (17-34 persons at imgsz 1600), but
4-corner calibration cannot be tapped when a corner is not visible.
Follow-ups queued for the calibration UI: pan/zoom in the corner picker,
and a "custom reference points" mode using any four known line
intersections (attack-line × sideline etc.) instead of court corners.

## Product posture (unchanged, now evidence-backed)

Near-half analytics (Mo's team when the camera is at her end) are the
trustworthy core; far-half output is presence/tendency grade near the net.
This matches D16/D18's attribution-free, confidence-gated design.
