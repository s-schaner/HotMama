import { useState } from "react";

import type { MatchStateDto, RosterPlayer } from "../types";

function jerseyLabel(player: RosterPlayer): string {
  return player.jersey === null ? player.name : `#${player.jersey}`;
}

/** Lineup picker: tap six players in serve order (P1 first), then start the set. */
export function SetControls({
  state,
  append,
}: {
  state: MatchStateDto;
  append: (event: Record<string, unknown>) => void;
}) {
  const [lineup, setLineup] = useState<string[]>([]);
  const [weServeFirst, setWeServeFirst] = useState(true);
  const nextSetNumber = state.sets.length + 1;
  const liberos = state.roster.filter((p) => p.is_libero).map((p) => p.player_id);

  if (state.match_over) {
    return (
      <div className="panel">
        <h3>Match complete</h3>
        <p className="subtle">
          {state.our_team} {state.sets_won_us} — {state.sets_won_them} {state.opponent}
        </p>
        {!state.session_closed && (
          <button onClick={() => append({ type: "session_closed" })}>Close session</button>
        )}
      </div>
    );
  }

  const toggle = (playerId: string) => {
    setLineup((current) =>
      current.includes(playerId)
        ? current.filter((id) => id !== playerId)
        : current.length < 6
          ? [...current, playerId]
          : current,
    );
  };

  const start = () => {
    append({
      type: "set_started",
      set_number: nextSetNumber,
      lineup,
      liberos,
      we_serve_first: weServeFirst,
    });
    setLineup([]);
  };

  return (
    <div className="panel">
      <h3>Start set {nextSetNumber} — tap starters in serve order</h3>
      <div className="lineup-slots" style={{ marginBottom: "0.7rem" }}>
        {Array.from({ length: 6 }, (_, slot) => {
          const playerId = lineup[slot];
          const player = state.roster.find((p) => p.player_id === playerId);
          return (
            <div key={slot} className={`lineup-slot ${player ? "filled" : ""}`}>
              <span className="p">P{slot + 1}</span>
              <span>{player ? jerseyLabel(player) : "—"}</span>
            </div>
          );
        })}
      </div>
      <div className="grid4" style={{ marginBottom: "0.7rem" }}>
        {state.roster
          .filter((p) => !p.is_libero)
          .map((player) => (
            <button
              key={player.player_id}
              className={`player-btn small ${lineup.includes(player.player_id) ? "selected" : ""}`}
              onClick={() => toggle(player.player_id)}
            >
              <span className="jersey">{player.jersey ?? "•"}</span>
              <span className="pname">{player.name}</span>
            </button>
          ))}
      </div>
      <div className="row">
        <button
          className={`small ${weServeFirst ? "us" : ""}`}
          onClick={() => setWeServeFirst(true)}
        >
          We serve first
        </button>
        <button
          className={`small ${!weServeFirst ? "them" : ""}`}
          onClick={() => setWeServeFirst(false)}
        >
          They serve first
        </button>
        <span className="spacer" style={{ flex: 1 }} />
        <button className="primary" disabled={lineup.length !== 6} onClick={start}>
          Start set
        </button>
      </div>
      {liberos.length > 0 && (
        <p className="subtle" style={{ marginBottom: 0 }}>
          Libero{liberos.length > 1 ? "s" : ""} available:{" "}
          {state.roster
            .filter((p) => p.is_libero)
            .map((p) => p.name)
            .join(", ")}
        </p>
      )}
    </div>
  );
}
