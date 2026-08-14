import { useState } from "react";

import { requestSummary } from "../api";
import { CalibrationPanel } from "../components/CalibrationPanel";
import { CapturePanel, ClipsPanel } from "../components/CapturePanel";
import { ProposalsPanel } from "../components/ProposalsPanel";
import { RotationStrip, rotationOf } from "../components/RotationStrip";
import { Scoreboard } from "../components/Scoreboard";
import { SetControls } from "../components/SetControls";
import type {
  CaptureStatusDto,
  ClipDto,
  RosterPlayer,
  SessionPayload,
  TagCode,
} from "../types";

const TAGS: { code: TagCode; label: string; good: boolean }[] = [
  { code: "great_serve", label: "🔥 Great serve", good: true },
  { code: "great_dig", label: "🛡 Great dig", good: true },
  { code: "attack_winner", label: "💥 Attack winner", good: true },
  { code: "highlight", label: "⭐ Flag moment", good: true },
  { code: "serve_error", label: "❌ Serve error", good: false },
  { code: "dig_failure", label: "❌ Dig failure", good: false },
  { code: "positioning_error", label: "❌ Positioning", good: false },
];

export function CoachView({
  payload,
  append,
  undo,
  notify,
  sessionId,
  actor,
  capture,
  onCaptureStatus,
  clips,
}: {
  payload: SessionPayload;
  append: (event: Record<string, unknown>) => void;
  undo: () => void;
  notify: (text: string) => void;
  sessionId: string;
  actor: string;
  capture: CaptureStatusDto;
  onCaptureStatus: (status: CaptureStatusDto) => void;
  clips: ClipDto[];
}) {
  const [tagPlayer, setTagPlayer] = useState<string | null>(null);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const state = payload.state;
  const summary = payload.summary;
  const set = state.current_set;
  const leak = summary.biggest_leak;

  const fetchSummary = async () => {
    setSummaryBusy(true);
    try {
      const result = await requestSummary(sessionId, actor);
      setAiSummary(result.summary);
    } catch (err) {
      notify(`⚠ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSummaryBusy(false);
    }
  };

  const tag = (code: TagCode) => {
    append({
      type: "moment_tagged",
      tag: code,
      ...(tagPlayer ? { player_id: tagPlayer } : {}),
    });
    notify("Tagged");
    setTagPlayer(null);
  };

  const serverPlayer: RosterPlayer | undefined = set
    ? state.roster.find((p) => p.player_id === set.our_server)
    : undefined;

  return (
    <>
      <Scoreboard state={state} />

      {leak && (
        <div className="banner warn leak" role="status">
          {leak.sentence}
        </div>
      )}

      <ProposalsPanel state={state} sessionId={sessionId} actor={actor} notify={notify} />

      {set && (
        <div className="panel">
          <h3>
            Rotation {set.our_rotation}
            {serverPlayer &&
              set.serving === "us" &&
              ` — ${serverPlayer.name} serving`}
          </h3>
          <div className="row subtle">
            {set.slot_owner.map((playerId, slot) => {
              const player = state.roster.find((p) => p.player_id === playerId);
              const floor = state.roster.find((p) => p.player_id === set.on_court[slot]);
              const showLibero = floor && floor.player_id !== playerId;
              return (
                <span key={slot}>
                  P{slot + 1} {player?.jersey != null ? `#${player.jersey}` : player?.name}
                  {showLibero ? ` (L:${floor.name})` : ""}
                </span>
              );
            })}
          </div>
        </div>
      )}

      <RotationStrip rows={summary.rotation_table} currentRotation={rotationOf(set)} />

      {!set && <SetControls state={state} append={append} />}

      {set && (
        <div className="panel">
          <h3>Tag a moment{tagPlayer ? " — player selected" : ""}</h3>
          <div className="grid2" style={{ marginBottom: "0.6rem" }}>
            {TAGS.map((entry) => (
              <button key={entry.code} className="small" onClick={() => tag(entry.code)}>
                {entry.label}
              </button>
            ))}
          </div>
          <div className="grid4">
            {set.on_court.map((playerId) => {
              const player = state.roster.find((p) => p.player_id === playerId);
              if (!player) return null;
              return (
                <button
                  key={playerId}
                  className={`small ghost ${tagPlayer === playerId ? "selected" : ""}`}
                  onClick={() => setTagPlayer(tagPlayer === playerId ? null : playerId)}
                >
                  {player.jersey != null ? `#${player.jersey}` : player.name}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {set && (
        <div className="panel">
          <h3>Coach controls</h3>
          <div className="grid4" style={{ marginBottom: "0.6rem" }}>
            <button
              className="small us"
              onClick={() => append({ type: "score_adjusted", us_delta: 1, note: "coach" })}
            >
              +1 us
            </button>
            <button
              className="small"
              onClick={() => append({ type: "score_adjusted", us_delta: -1, note: "coach" })}
            >
              −1 us
            </button>
            <button
              className="small them"
              onClick={() => append({ type: "score_adjusted", them_delta: 1, note: "coach" })}
            >
              +1 them
            </button>
            <button
              className="small"
              onClick={() => append({ type: "score_adjusted", them_delta: -1, note: "coach" })}
            >
              −1 them
            </button>
          </div>
          <div className="grid4">
            <button
              className="small"
              onClick={() => append({ type: "timeout_called", team: "us" })}
            >
              ⏱ TO us ({set.timeouts.us ?? 0})
            </button>
            <button
              className="small"
              onClick={() => append({ type: "timeout_called", team: "them" })}
            >
              ⏱ TO them ({set.timeouts.them ?? 0})
            </button>
            <button className="small ghost" onClick={undo}>
              ↩ Undo
            </button>
            <button
              className="small danger"
              disabled={set.us_points === set.them_points}
              onClick={() => append({ type: "set_ended" })}
            >
              End set
            </button>
          </div>
        </div>
      )}

      <PersonnelPanel payload={payload} append={append} />

      <CapturePanel
        sessionId={sessionId}
        status={capture}
        onStatus={onCaptureStatus}
        notify={notify}
      />
      <CalibrationPanel
        sessionId={sessionId}
        recording={capture.state === "recording"}
        notify={notify}
        onSaved={() => undefined}
      />
      <ClipsPanel clips={clips} />

      <div className="row">
        <button
          className="small ghost"
          onClick={() => window.open(`/api/sessions/${sessionId}/report`, "_blank")}
        >
          📄 Session report
        </button>
        <button
          className="small ghost"
          onClick={() => window.open(`/api/sessions/${sessionId}/report.pdf`, "_blank")}
        >
          ⬇ PDF
        </button>
        <button className="small ghost" disabled={summaryBusy} onClick={fetchSummary}>
          {summaryBusy ? "🧠 Thinking…" : "🧠 Set summary"}
        </button>
        {state.cv_observations > 0 && (
          <span className="subtle">
            🤖 {state.cv_observations} observations from remote workers
          </span>
        )}
      </div>

      {aiSummary && (
        <div className="panel">
          <h3>Between-set read</h3>
          <p style={{ whiteSpace: "pre-wrap", margin: 0 }}>{aiSummary}</p>
        </div>
      )}

      {state.warnings.length > 0 && (
        <div className="panel">
          <h3>Warnings</h3>
          {state.warnings.map((warning, index) => (
            <div key={index} className="subtle">
              {warning}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function PersonnelPanel({
  payload,
  append,
}: {
  payload: SessionPayload;
  append: (event: Record<string, unknown>) => void;
}) {
  const state = payload.state;
  const set = state.current_set;
  const [playerIn, setPlayerIn] = useState("");
  const [playerOut, setPlayerOut] = useState("");
  if (!set) return null;

  const onFloor = new Set(set.on_court);
  const bench = state.roster.filter(
    (p) => !onFloor.has(p.player_id) && !set.liberos.includes(p.player_id),
  );
  const liberosOff = state.roster.filter(
    (p) => set.liberos.includes(p.player_id) && !onFloor.has(p.player_id),
  );
  const candidatesIn = [...bench, ...liberosOff];
  const isLiberoMove =
    set.liberos.includes(playerIn) ||
    set.liberos.includes(playerOut);

  const commit = () => {
    if (!playerIn || !playerOut) return;
    append({
      type: isLiberoMove ? "libero_swap" : "sub_made",
      player_in: playerIn,
      player_out: playerOut,
    });
    setPlayerIn("");
    setPlayerOut("");
  };

  return (
    <div className="panel">
      <h3>Personnel (subs: {set.subs_used})</h3>
      <div className="row">
        <select value={playerIn} onChange={(e) => setPlayerIn(e.target.value)} style={{ flex: 1 }}>
          <option value="">In…</option>
          {candidatesIn.map((p) => (
            <option key={p.player_id} value={p.player_id}>
              {p.jersey != null ? `#${p.jersey} ` : ""}
              {p.name}
              {p.is_libero ? " (L)" : ""}
            </option>
          ))}
        </select>
        <select
          value={playerOut}
          onChange={(e) => setPlayerOut(e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="">Out…</option>
          {set.on_court.map((playerId) => {
            const player = state.roster.find((p) => p.player_id === playerId);
            if (!player) return null;
            return (
              <option key={playerId} value={playerId}>
                {player.jersey != null ? `#${player.jersey} ` : ""}
                {player.name}
              </option>
            );
          })}
        </select>
        <button className="small" disabled={!playerIn || !playerOut} onClick={commit}>
          {isLiberoMove ? "Libero swap" : "Sub"}
        </button>
      </div>
    </div>
  );
}
