import type { RotationRow, SetDto } from "../types";

function pct(value: number | null): string {
  if (value === null) return "—";
  return `${Math.round(value * 100)}%`;
}

export function RotationStrip({
  rows,
  currentRotation,
}: {
  rows: RotationRow[];
  currentRotation: number | null;
}) {
  return (
    <div className="panel">
      <h3>Side-out % by rotation</h3>
      <div className="rot-strip">
        {rows.map((row, index) => {
          const rotation = row.rotation ?? index + 1;
          return (
            <div
              key={rotation}
              className={`rot-tile ${rotation === currentRotation ? "current" : ""}`}
            >
              <div className="r">R{rotation}</div>
              <div className="v">{pct(row.side_out_pct)}</div>
              <div
                className={`d ${row.point_diff > 0 ? "pos" : row.point_diff < 0 ? "neg" : ""}`}
              >
                {row.point_diff > 0 ? `+${row.point_diff}` : row.point_diff}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function rotationOf(set: SetDto | null): number | null {
  return set ? set.our_rotation : null;
}
