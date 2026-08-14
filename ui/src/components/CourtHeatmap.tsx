import type { MatchStateDto } from "../types";

/** Position density on a court diagram, fed by remote-worker tracking. */
export function CourtHeatmap({ state }: { state: MatchStateDto }) {
  const heatmap = state.court_heatmap;
  if (!heatmap || heatmap.observations === 0) return null;
  const { cols, rows, cells } = heatmap;
  const peak = Math.max(...cells, 1);
  const ourSide = state.current_set?.our_side ?? null;

  // Render far half at the top (rows are stored near-first).
  const displayRows = Array.from({ length: rows }, (_, viewRow) => {
    const dataRow = rows - 1 - viewRow;
    return cells.slice(dataRow * cols, (dataRow + 1) * cols);
  });

  return (
    <div className="panel">
      <h3>
        Court coverage · {heatmap.observations} analyzed rall
        {heatmap.observations === 1 ? "y" : "ies"}
      </h3>
      <div className="subtle" style={{ marginBottom: "0.4rem" }}>
        FAR {ourSide === "far" ? `— ${state.our_team}` : ourSide === "near" ? `— ${state.opponent}` : ""}
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gap: 2,
          borderRadius: 10,
          overflow: "hidden",
        }}
      >
        {displayRows.map((rowCells, viewRow) =>
          rowCells.map((value, col) => {
            const heat = value / peak;
            const isNetBorder = viewRow === rows / 2;
            return (
              <div
                key={`${viewRow}-${col}`}
                style={{
                  aspectRatio: "3 / 2",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.7rem",
                  fontWeight: 700,
                  color: heat > 0.4 ? "#16100c" : "var(--muted)",
                  background: `rgba(255, 122, 69, ${0.06 + heat * 0.9})`,
                  borderTop: isNetBorder ? "3px solid var(--warn)" : "none",
                }}
              >
                {value > 0 ? value : ""}
              </div>
            );
          }),
        )}
      </div>
      <div className="subtle" style={{ marginTop: "0.4rem" }}>
        NEAR (camera end){" "}
        {ourSide === "near" ? `— ${state.our_team}` : ourSide === "far" ? `— ${state.opponent}` : ""}
        {" · "}
        {heatmap.near_hits + heatmap.far_hits} on-court positions,{" "}
        {heatmap.out_of_bounds_hits} bystanders filtered
      </div>
    </div>
  );
}
