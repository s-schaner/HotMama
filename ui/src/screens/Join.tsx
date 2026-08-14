import { useEffect, useState } from "react";

import { createSession, listSessions } from "../api";
import type { Role, SessionListItem } from "../types";

/**
 * Roster entry format, one player per line: `Name, jersey[, L]`.
 * Example:
 *   Maya, 7
 *   Jo, 12, L
 */
function parseRoster(text: string) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const parts = line.split(",").map((part) => part.trim());
      const name = parts[0] || `Player ${index + 1}`;
      const jersey = parts[1] !== undefined && parts[1] !== "" ? Number(parts[1]) : null;
      const isLibero = (parts[2] ?? "").toUpperCase() === "L";
      return {
        player_id: `p${index + 1}`,
        name,
        jersey: Number.isFinite(jersey as number) ? jersey : null,
        is_libero: isLibero,
      };
    });
}

export function Join({ onEnter }: { onEnter: (sessionId: string, role: Role) => void }) {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [role, setRole] = useState<Role>("coach");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [ourTeam, setOurTeam] = useState("HotMama");
  const [opponent, setOpponent] = useState("Opponent");
  const [kind, setKind] = useState("match");
  const [bestOf, setBestOf] = useState(5);
  const [rosterText, setRosterText] = useState("");

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch(() => setError("Can't reach the court host — is the server running?"));
  }, []);

  const create = async () => {
    setCreating(true);
    setError(null);
    try {
      const roster = parseRoster(rosterText);
      if (roster.length < 6) {
        setError("Need at least 6 players (one per line: Name, jersey[, L])");
        return;
      }
      const result = await createSession({
        kind,
        our_team: ourTeam,
        opponent,
        best_of: bestOf,
        roster,
      });
      onEnter(result.session_id, role);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">HOTMAMA</span>
        <span className="spacer" />
        <span className="subtle">v2</span>
      </header>
      <main className="main">
        <div className="panel">
          <h3>Join as</h3>
          <div className="grid2">
            <button
              className={role === "coach" ? "selected" : ""}
              onClick={() => setRole("coach")}
            >
              🧠 Coach
            </button>
            <button
              className={role === "statter" ? "selected" : ""}
              onClick={() => setRole("statter")}
            >
              ✍️ Statter
            </button>
          </div>
        </div>

        {sessions.length > 0 && (
          <div className="panel">
            <h3>Resume a session</h3>
            <div className="join-list">
              {sessions.map((session) => (
                <button
                  key={session.session_id}
                  className="join-item small"
                  onClick={() => onEnter(session.session_id, role)}
                >
                  <span className="label">{session.label || session.session_id}</span>
                  <span className="meta">
                    {session.kind} · {session.last_seq} events
                    {session.closed ? " · closed" : ""}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="panel">
          <h3>New session</h3>
          <div className="grid2" style={{ marginBottom: "0.6rem" }}>
            <label className="field">
              Our team
              <input value={ourTeam} onChange={(e) => setOurTeam(e.target.value)} />
            </label>
            <label className="field">
              Opponent
              <input value={opponent} onChange={(e) => setOpponent(e.target.value)} />
            </label>
            <label className="field">
              Type
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                <option value="match">Match</option>
                <option value="practice">Practice</option>
                <option value="scrimmage">Scrimmage</option>
              </select>
            </label>
            <label className="field">
              Format
              <select value={bestOf} onChange={(e) => setBestOf(Number(e.target.value))}>
                <option value={5}>Best of 5</option>
                <option value={3}>Best of 3</option>
                <option value={1}>Single set</option>
              </select>
            </label>
          </div>
          <label className="field" style={{ marginBottom: "0.6rem" }}>
            Roster — one per line: Name, jersey (add “, L” for libero)
            <textarea
              value={rosterText}
              onChange={(e) => setRosterText(e.target.value)}
              placeholder={"Maya, 7\nJo, 12\nSam, 3\nAlex, 9\nRiley, 5\nCasey, 11\nDrew, 2, L"}
            />
          </label>
          {error && <div className="banner warn" style={{ marginBottom: "0.6rem" }}>{error}</div>}
          <button className="primary" disabled={creating} onClick={create}>
            {creating ? "Creating…" : "Create & join"}
          </button>
        </div>
      </main>
    </div>
  );
}
