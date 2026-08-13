import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { SessionSocket, type SocketStatus } from "./api";
import { CoachView } from "./screens/CoachView";
import { Join } from "./screens/Join";
import { StatterView } from "./screens/StatterView";
import type { Role, ServerMessage, SessionPayload } from "./types";

interface Active {
  sessionId: string;
  role: Role;
}

function deviceName(): string {
  const stored = localStorage.getItem("hotmama.device");
  if (stored) return stored;
  const name = `device-${Math.random().toString(36).slice(2, 7)}`;
  localStorage.setItem("hotmama.device", name);
  return name;
}

export function App() {
  const [active, setActive] = useState<Active | null>(() => {
    const raw = localStorage.getItem("hotmama.active");
    return raw ? (JSON.parse(raw) as Active) : null;
  });

  const enter = useCallback((sessionId: string, role: Role) => {
    const next = { sessionId, role };
    localStorage.setItem("hotmama.active", JSON.stringify(next));
    setActive(next);
  }, []);

  const leave = useCallback(() => {
    localStorage.removeItem("hotmama.active");
    setActive(null);
  }, []);

  if (!active) return <Join onEnter={enter} />;
  return <Session active={active} onLeave={leave} onSwitchRole={enter} />;
}

function Session({
  active,
  onLeave,
  onSwitchRole,
}: {
  active: Active;
  onLeave: () => void;
  onSwitchRole: (sessionId: string, role: Role) => void;
}) {
  const [payload, setPayload] = useState<SessionPayload | null>(null);
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const [toast, setToast] = useState<string | null>(null);
  const socketRef = useRef<SessionSocket | null>(null);
  const actor = useMemo(() => `${deviceName()}:${active.role}`, [active.role]);

  const showToast = useCallback((text: string) => {
    setToast(text);
    window.setTimeout(() => setToast(null), 2500);
  }, []);

  useEffect(() => {
    const socket = new SessionSocket(
      active.sessionId,
      (message: ServerMessage) => {
        if (message.type === "snapshot" || message.type === "event") {
          setPayload(message);
        } else if (message.type === "error") {
          showToast(`⚠ ${message.detail}`);
        }
      },
      setStatus,
    );
    socketRef.current = socket;
    socket.connect();
    return () => socket.close();
  }, [active.sessionId, showToast]);

  const append = useCallback(
    (event: Record<string, unknown>) => {
      socketRef.current?.appendEvent(event, actor);
    },
    [actor],
  );

  const undo = useCallback(() => {
    socketRef.current?.undo(actor);
    showToast("Undone");
  }, [actor, showToast]);

  const otherRole: Role = active.role === "coach" ? "statter" : "coach";

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">HOTMAMA</span>
        <span className={`conn ${status}`} title={status} />
        <span className="spacer" />
        <button
          className="small ghost"
          onClick={() => onSwitchRole(active.sessionId, otherRole)}
        >
          {active.role === "coach" ? "→ Statter" : "→ Coach"}
        </button>
        <button className="small ghost" onClick={onLeave}>
          Exit
        </button>
      </header>
      <main className="main">
        {!payload ? (
          <div className="panel">Connecting to session…</div>
        ) : active.role === "coach" ? (
          <CoachView payload={payload} append={append} undo={undo} notify={showToast} />
        ) : (
          <StatterView payload={payload} append={append} undo={undo} notify={showToast} />
        )}
      </main>
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
