import { useState } from "react";

import { startCapture, stopCapture } from "../api";
import type { CaptureStatusDto, ClipDto } from "../types";

export function CapturePanel({
  sessionId,
  status,
  onStatus,
  notify,
}: {
  sessionId: string;
  status: CaptureStatusDto;
  onStatus: (status: CaptureStatusDto) => void;
  notify: (text: string) => void;
}) {
  const [source, setSource] = useState(
    () => localStorage.getItem("hotmama.capture.source") ?? "0",
  );
  const [busy, setBusy] = useState(false);
  const recording = status.state === "recording";

  const start = async () => {
    setBusy(true);
    try {
      localStorage.setItem("hotmama.capture.source", source);
      onStatus(await startCapture(sessionId, source));
      notify("Recording started");
    } catch (err) {
      notify(`⚠ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      onStatus(await stopCapture(sessionId));
      notify("Recording stopped");
    } catch (err) {
      notify(`⚠ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h3>
        Camera{" "}
        <span className={`conn ${recording ? "open" : status.state === "error" ? "" : "connecting"}`} />
      </h3>
      <div className="row" style={{ marginBottom: "0.5rem" }}>
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder='USB index ("0"), rtsp:// URL, or file path'
          disabled={recording}
          style={{ flex: 1 }}
        />
        {recording ? (
          <button className="small danger" disabled={busy} onClick={stop}>
            ■ Stop
          </button>
        ) : (
          <button className="small primary" disabled={busy || !source.trim()} onClick={start}>
            ● Record
          </button>
        )}
      </div>
      <div className="subtle">
        {status.state === "recording" &&
          `${status.codec ?? "…"} · ${status.width}×${status.height} @ ${Math.round(
            status.fps ?? 0,
          )}fps · seg ${status.segments ?? 0} · ${status.frames_total ?? 0} frames`}
        {status.state === "idle" && "Not recording. Tag clips and rally chunks need a camera."}
        {status.state === "finished" && "Recording finished — clips remain available below."}
        {status.state === "error" && `Camera error: ${status.error ?? "unknown"}`}
      </div>
    </div>
  );
}

function clipTitle(clip: ClipDto): string {
  return clip.kind === "tag" ? `🏷 ${clip.label.replace(/_/g, " ")}` : `🏐 ${clip.label}`;
}

export function ClipsPanel({ clips }: { clips: ClipDto[] }) {
  const [openClip, setOpenClip] = useState<string | null>(null);
  const [showRallies, setShowRallies] = useState(false);
  const visible = clips.filter((clip) => showRallies || clip.kind === "tag");

  if (clips.length === 0) return null;
  const rallyCount = clips.filter((c) => c.kind === "rally").length;

  return (
    <div className="panel">
      <h3>
        Clips ({visible.length})
        {rallyCount > 0 && (
          <button
            className="small ghost"
            style={{ marginLeft: "0.6rem" }}
            onClick={() => setShowRallies(!showRallies)}
          >
            {showRallies ? "Hide" : "Show"} rally chunks ({rallyCount})
          </button>
        )}
      </h3>
      <div className="join-list">
        {visible.map((clip) => (
          <div key={clip.clip_id}>
            <button
              className="join-item small"
              onClick={() =>
                setOpenClip(openClip === clip.clip_id ? null : clip.clip_id)
              }
              disabled={clip.status !== "ready"}
            >
              <span className="label">{clipTitle(clip)}</span>
              <span className="meta">
                {clip.status === "pending" && "⏳ cutting…"}
                {clip.status === "ready" && new Date(clip.start_at).toLocaleTimeString()}
                {clip.status === "failed" && `✗ ${clip.error ?? "failed"}`}
              </span>
            </button>
            {openClip === clip.clip_id && clip.url && (
              <video
                controls
                autoPlay
                playsInline
                preload="metadata"
                src={clip.url}
                style={{ width: "100%", borderRadius: 10, marginTop: "0.4rem" }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
