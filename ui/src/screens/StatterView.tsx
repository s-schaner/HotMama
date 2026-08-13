import { useState } from "react";

import { SetControls } from "../components/SetControls";
import { Scoreboard } from "../components/Scoreboard";
import type {
  MatchStateDto,
  PointReason,
  RosterPlayer,
  SessionPayload,
  Team,
} from "../types";

type Stage = { step: "outcome" } | { step: "reason"; winner: Team } | {
  step: "player";
  winner: Team;
  reason: PointReason;
};

interface ReasonOption {
  code: PointReason;
  label: string;
}

// KILL/ACE/BLOCK credit the winner; ERR_* charge the loser (see PointReason docs).
function reasonOptions(winner: Team, state: MatchStateDto): ReasonOption[] {
  const opponentName = winner === "us" ? state.opponent : state.our_team;
  const winnerErrPrefix = winner === "us" ? `${opponentName}` : "Our";
  if (winner === "us") {
    return [
      { code: "kill", label: "Kill" },
      { code: "ace", label: "Ace" },
      { code: "block", label: "Block" },
      { code: "err_serve", label: `${winnerErrPrefix} serve error` },
      { code: "err_attack", label: `${winnerErrPrefix} attack error` },
      { code: "err_net", label: `${winnerErrPrefix} net fault` },
      { code: "err_handling", label: `${winnerErrPrefix} ball handling` },
      { code: "err_other", label: `${winnerErrPrefix} other fault` },
    ];
  }
  return [
    { code: "kill", label: "Their kill" },
    { code: "ace", label: "Ace against us" },
    { code: "block", label: "We got blocked" },
    { code: "err_serve", label: "Our serve error" },
    { code: "err_attack", label: "Our attack error" },
    { code: "err_net", label: "Our net fault" },
    { code: "err_handling", label: "Our ball handling" },
    { code: "err_other", label: "Our other fault" },
  ];
}

// Attribute a player when the reason involves OUR team's action.
function playerRelevant(winner: Team, reason: PointReason): boolean {
  if (winner === "us") return ["kill", "ace", "block"].includes(reason);
  return reason.startsWith("err_");
}

export function StatterView({
  payload,
  append,
  undo,
  notify,
}: {
  payload: SessionPayload;
  append: (event: Record<string, unknown>) => void;
  undo: () => void;
  notify: (text: string) => void;
}) {
  const [stage, setStage] = useState<Stage>({ step: "outcome" });
  const state = payload.state;
  const set = state.current_set;

  const commit = (winner: Team, reason: PointReason, playerId: string | null) => {
    append({
      type: "rally_ended",
      winner,
      reason,
      ...(playerId ? { player_id: playerId } : {}),
    });
    notify(winner === "us" ? `Point ${state.our_team}` : `Point ${state.opponent}`);
    setStage({ step: "outcome" });
  };

  if (!set) {
    return (
      <>
        <Scoreboard state={state} />
        <SetControls state={state} append={append} />
      </>
    );
  }

  const onCourt = set.on_court
    .map((id) => state.roster.find((p) => p.player_id === id))
    .filter((p): p is RosterPlayer => Boolean(p));

  return (
    <>
      <Scoreboard state={state} />

      {set.decided ? (
        <div className="panel flow-stage">
          <button className="primary" onClick={() => append({ type: "set_ended" })}>
            Confirm end of set {set.set_number}
          </button>
          <button className="small ghost" onClick={undo}>
            ↩ Undo last point
          </button>
        </div>
      ) : stage.step === "outcome" ? (
        <div className="panel flow-stage">
          <div className="grid2">
            <button className="big-point us" onClick={() => setStage({ step: "reason", winner: "us" })}>
              POINT {state.our_team.toUpperCase()}
            </button>
            <button
              className="big-point them"
              onClick={() => setStage({ step: "reason", winner: "them" })}
            >
              POINT {state.opponent.toUpperCase()}
            </button>
          </div>
          <div className="row">
            <button className="small ghost" onClick={undo}>
              ↩ Undo
            </button>
            <span className="crumb">
              R{set.our_rotation} · serving: {set.serving === "us" ? state.our_team : state.opponent}
            </span>
          </div>
        </div>
      ) : stage.step === "reason" ? (
        <div className="panel flow-stage">
          <span className="crumb">
            Point {stage.winner === "us" ? state.our_team : state.opponent} — how?
          </span>
          <div className="grid2">
            {reasonOptions(stage.winner, state).map((option) => (
              <button
                key={option.code}
                className="reason-btn"
                onClick={() => {
                  if (playerRelevant(stage.winner, option.code)) {
                    setStage({ step: "player", winner: stage.winner, reason: option.code });
                  } else {
                    commit(stage.winner, option.code, null);
                  }
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="row">
            <button className="small ghost" onClick={() => commit(stage.winner, "unknown", null)}>
              Skip — just the point
            </button>
            <button className="small ghost" onClick={() => setStage({ step: "outcome" })}>
              ← Back
            </button>
          </div>
        </div>
      ) : (
        <div className="panel flow-stage">
          <span className="crumb">Who?</span>
          <div className="grid3">
            {onCourt.map((player) => (
              <button
                key={player.player_id}
                className="player-btn"
                onClick={() => commit(stage.winner, stage.reason, player.player_id)}
              >
                <span className="jersey">{player.jersey ?? "•"}</span>
                <span className="pname">{player.name}</span>
              </button>
            ))}
          </div>
          <div className="row">
            <button className="small ghost" onClick={() => commit(stage.winner, stage.reason, null)}>
              Skip player
            </button>
            <button
              className="small ghost"
              onClick={() => setStage({ step: "reason", winner: stage.winner })}
            >
              ← Back
            </button>
          </div>
        </div>
      )}

      <div className="panel">
        <h3>Last points</h3>
        <div className="points-list">
          {set.points
            .slice(-5)
            .reverse()
            .map((point) => (
              <div key={point.event_id} className="pt">
                <span className={`who ${point.winner}`}>
                  {point.winner === "us" ? state.our_team : state.opponent}
                </span>
                <span>
                  {point.us_points}-{point.them_points} · R{point.our_rotation} ·{" "}
                  {point.reason.replace("err_", "error: ")}
                </span>
              </div>
            ))}
        </div>
      </div>
    </>
  );
}
