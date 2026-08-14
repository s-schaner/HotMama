import type { MatchStateDto } from "../types";

export function Scoreboard({ state }: { state: MatchStateDto }) {
  const set = state.current_set;
  const lastFinished = state.sets.filter((s) => s.finished).at(-1);
  const us = set ? set.us_points : lastFinished?.us_points ?? 0;
  const them = set ? set.them_points : lastFinished?.them_points ?? 0;

  return (
    <div className="panel">
      <div className="scoreboard">
        <div className="team">
          <span className="name">
            {state.our_team}
            {set?.serving === "us" && <span className="serve-dot" />}
          </span>
          <span className="pts us">{us}</span>
        </div>
        <div className="mid">
          <span>
            {set
              ? `SET ${set.set_number} · to ${set.to_win}`
              : state.match_over
                ? "FINAL"
                : "BETWEEN SETS"}
          </span>
          <div className="pips">
            {state.sets.map((s) => (
              <span
                key={s.set_number}
                className={`pip ${
                  s.won_by === "us" ? "won-us" : s.won_by === "them" ? "won-them" : ""
                }`}
              />
            ))}
          </div>
          <span>
            {state.sets_won_us} — {state.sets_won_them}
          </span>
        </div>
        <div className="team">
          <span className="name">
            {state.opponent}
            {set?.serving === "them" && <span className="serve-dot" />}
          </span>
          <span className="pts them">{them}</span>
        </div>
      </div>
      {set?.set_point && (
        <div className="banner warn" style={{ marginTop: "0.6rem" }}>
          SET POINT — {set.set_point === "us" ? state.our_team : state.opponent}
        </div>
      )}
      {set?.decided && (
        <div className="banner info" style={{ marginTop: "0.6rem" }}>
          Set decided for {set.decided === "us" ? state.our_team : state.opponent} — confirm
          end of set
        </div>
      )}
    </div>
  );
}
