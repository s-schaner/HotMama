import { useRef, useState } from "react";

import { listVenues, saveVenue } from "../api";

type Mode = "near_half" | "full";

const CORNER_LABELS: Record<Mode, string[]> = {
  near_half: ["near-LEFT baseline", "near-RIGHT baseline", "net RIGHT", "net LEFT"],
  full: ["near-LEFT baseline", "near-RIGHT baseline", "far-RIGHT baseline", "far-LEFT baseline"],
};

/** Tap the court corners on a live frame; save as a named venue (once a season). */
export function CalibrationPanel({
  sessionId,
  recording,
  notify,
  onSaved,
}: {
  sessionId: string;
  recording: boolean;
  notify: (text: string) => void;
  onSaved: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [points, setPoints] = useState<[number, number][]>([]);
  const [mode, setMode] = useState<Mode>("near_half");
  const [venueName, setVenueName] = useState("");
  const [busy, setBusy] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const loadFrame = () => {
    setPoints([]);
    setFrameUrl(`/api/sessions/${sessionId}/capture/frame?t=${Date.now()}`);
  };

  const onImageClick = (event: React.MouseEvent<HTMLImageElement>) => {
    const img = imgRef.current;
    if (!img || points.length >= 4) return;
    const rect = img.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * img.naturalWidth;
    const y = ((event.clientY - rect.top) / rect.height) * img.naturalHeight;
    setPoints([...points, [Math.round(x * 10) / 10, Math.round(y * 10) / 10]]);
  };

  const save = async () => {
    const img = imgRef.current;
    if (!img || points.length !== 4 || !venueName.trim()) return;
    setBusy(true);
    try {
      await saveVenue(venueName.trim(), {
        image_corners: points,
        frame_width: img.naturalWidth,
        frame_height: img.naturalHeight,
        mode,
      });
      notify(`Venue “${venueName.trim()}” calibrated`);
      onSaved();
      setPoints([]);
    } catch (err) {
      notify(`⚠ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  if (!open) {
    return (
      <div className="panel">
        <h3>Venue calibration</h3>
        <button className="small ghost" onClick={() => setOpen(true)}>
          🎯 Calibrate this court
        </button>
      </div>
    );
  }

  const nextLabel = CORNER_LABELS[mode][points.length];

  return (
    <div className="panel">
      <h3>Venue calibration</h3>
      <div className="row" style={{ marginBottom: "0.5rem" }}>
        <button
          className={`small ${mode === "near_half" ? "us" : "ghost"}`}
          onClick={() => {
            setMode("near_half");
            setPoints([]);
          }}
        >
          Near half + net (recommended)
        </button>
        <button
          className={`small ${mode === "full" ? "us" : "ghost"}`}
          onClick={() => {
            setMode("full");
            setPoints([]);
          }}
        >
          Full court corners
        </button>
        <button className="small ghost" disabled={!recording} onClick={loadFrame}>
          📷 {frameUrl ? "Refresh frame" : "Grab live frame"}
        </button>
      </div>
      {!recording && !frameUrl && (
        <p className="subtle">Start the camera first, then grab a frame to tap on.</p>
      )}
      {frameUrl && (
        <>
          <p className="subtle" style={{ marginTop: 0 }}>
            {points.length < 4 ? `Tap the ${nextLabel} corner` : "All 4 corners set"}
            {points.length > 0 && (
              <button
                className="small ghost"
                style={{ marginLeft: "0.6rem" }}
                onClick={() => setPoints([])}
              >
                Reset
              </button>
            )}
          </p>
          <div style={{ position: "relative", marginBottom: "0.6rem" }}>
            <img
              ref={imgRef}
              src={frameUrl}
              onClick={onImageClick}
              style={{ width: "100%", borderRadius: 10, cursor: "crosshair" }}
              alt="live court frame"
            />
            {points.map(([x, y], index) => {
              const img = imgRef.current;
              if (!img) return null;
              return (
                <span
                  key={index}
                  style={{
                    position: "absolute",
                    left: `${(x / img.naturalWidth) * 100}%`,
                    top: `${(y / img.naturalHeight) * 100}%`,
                    transform: "translate(-50%, -50%)",
                    width: 16,
                    height: 16,
                    borderRadius: "50%",
                    background: "var(--good)",
                    border: "2px solid #fff",
                    color: "#0b0e11",
                    fontSize: 10,
                    fontWeight: 800,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {index + 1}
                </span>
              );
            })}
          </div>
          <div className="row">
            <input
              value={venueName}
              onChange={(e) => setVenueName(e.target.value)}
              placeholder="Venue name (e.g. Lincoln HS main gym)"
              style={{ flex: 1 }}
            />
            <button
              className="small primary"
              disabled={busy || points.length !== 4 || !venueName.trim()}
              onClick={save}
            >
              {busy ? "Saving…" : "💾 Save venue"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export { listVenues };
