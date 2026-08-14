import { useEffect, useState } from "react";

import {
  createSession,
  listRosters,
  listSessions,
  listVenues,
  saveRoster,
  type SavedRoster,
  type Venue,
} from "../api";
import type { Role, SessionListItem } from "../types";

/** Average-size squad, ready to edit: 12 players, #12 as libero. */
const DEFAULT_TEMPLATE = Array.from({ length: 12 }, (_, index) => ({
  player_id: `p${index + 1}`,
  name: `Player ${index + 1}`,
  jersey: index + 1,
  is_libero: index === 11,
}));

function rosterToText(players: SavedRoster["players"]): string {
  return players
    .map(
      (player) =>
        `${player.name}, ${player.jersey ?? ""}${player.is_libero ? ", L" : ""}`,
    )
    .join("\n");
}

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
  const [savedRosters, setSavedRosters] = useState<SavedRoster[]>([]);
  const [rosterName, setRosterName] = useState("");
  const [savingRoster, setSavingRoster] = useState(false);
  const [venues, setVenues] = useState<Venue[]>([]);
  const [venue, setVenue] = useState("");

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch(() => setError("Can't reach the court host — is the server running?"));
    listRosters()
      .then(setSavedRosters)
      .catch(() => undefined);
    listVenues()
      .then(setVenues)
      .catch(() => undefined);
  }, []);

  const loadTemplate = (value: string) => {
    if (value === "default") {
      setRosterText(rosterToText(DEFAULT_TEMPLATE));
      setRosterName("");
      return;
    }
    const saved = savedRosters.find((roster) => roster.name === value);
    if (saved) {
      setRosterText(rosterToText(saved.players));
      setRosterName(saved.name);
    }
  };

  const persistRoster = async () => {
    const players = parseRoster(rosterText);
    if (players.length < 6) {
      setError("Need at least 6 players before saving a roster");
      return;
    }
    if (!rosterName.trim()) {
      setError("Give the roster a name to save it");
      return;
    }
    setSavingRoster(true);
    setError(null);
    try {
      await saveRoster(rosterName.trim(), players);
      setSavedRosters(await listRosters());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingRoster(false);
    }
  };

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
        venue: venue || null,
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
            <label className="field">
              Venue (calibration)
              <select value={venue} onChange={(e) => setVenue(e.target.value)}>
                <option value="">None yet — calibrate in-session</option>
                {venues.map((entry) => (
                  <option key={entry.name} value={entry.name}>
                    🎯 {entry.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="field" style={{ marginBottom: "0.5rem" }}>
            Start from
            <select defaultValue="" onChange={(e) => loadTemplate(e.target.value)}>
              <option value="" disabled>
                Pick a template or saved roster…
              </option>
              <option value="default">Default squad (12 players)</option>
              {savedRosters.map((roster) => (
                <option key={roster.name} value={roster.name}>
                  💾 {roster.name} ({roster.players.length})
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ marginBottom: "0.5rem" }}>
            Roster — one per line: Name, jersey (add “, L” for libero)
            <textarea
              value={rosterText}
              onChange={(e) => setRosterText(e.target.value)}
              placeholder={"Maya, 7\nJo, 12\nSam, 3\nAlex, 9\nRiley, 5\nCasey, 11\nDrew, 2, L"}
            />
          </label>
          <div className="row" style={{ marginBottom: "0.6rem" }}>
            <input
              value={rosterName}
              onChange={(e) => setRosterName(e.target.value)}
              placeholder="Roster name (e.g. HotMama 2026)"
              style={{ flex: 1 }}
            />
            <button
              className="small"
              disabled={savingRoster || !rosterText.trim()}
              onClick={persistRoster}
            >
              {savingRoster ? "Saving…" : "💾 Save roster"}
            </button>
          </div>
          {error && <div className="banner warn" style={{ marginBottom: "0.6rem" }}>{error}</div>}
          <button className="primary" disabled={creating} onClick={create}>
            {creating ? "Creating…" : "Create & join"}
          </button>
        </div>
      </main>
    </div>
  );
}
