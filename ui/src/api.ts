// REST + WebSocket client. Same-origin in production (the host serves the PWA);
// the Vite dev server proxies /api and /ws to the host during development.

import type { ServerMessage, SessionListItem, SessionPayload } from "./types";

export async function listSessions(): Promise<SessionListItem[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error(`list sessions failed: ${res.status}`);
  return res.json();
}

export interface CreateSessionBody {
  label?: string;
  kind: string;
  our_team: string;
  opponent: string;
  best_of: number;
  roster: {
    player_id: string;
    name: string;
    jersey: number | null;
    is_libero: boolean;
  }[];
  actor?: string;
}

export async function createSession(
  body: CreateSessionBody,
): Promise<SessionPayload & { session_id: string }> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`create session failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export type SocketStatus = "connecting" | "open" | "closed";

export class SessionSocket {
  private ws: WebSocket | null = null;
  private closedByUser = false;
  private retryDelay = 500;
  private pingTimer: number | null = null;

  constructor(
    private readonly sessionId: string,
    private readonly onMessage: (message: ServerMessage) => void,
    private readonly onStatus: (status: SocketStatus) => void,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.open();
  }

  private open(): void {
    this.onStatus("connecting");
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${location.host}/ws/sessions/${this.sessionId}`);

    this.ws.onopen = () => {
      this.retryDelay = 500;
      this.onStatus("open");
      this.pingTimer = window.setInterval(() => this.send({ type: "ping" }), 20000);
    };
    this.ws.onmessage = (raw) => {
      try {
        this.onMessage(JSON.parse(raw.data as string) as ServerMessage);
      } catch {
        // Malformed frames are dropped; the next snapshot resyncs everything.
      }
    };
    this.ws.onclose = () => {
      if (this.pingTimer !== null) window.clearInterval(this.pingTimer);
      this.pingTimer = null;
      this.onStatus("closed");
      if (!this.closedByUser) {
        window.setTimeout(() => this.open(), this.retryDelay);
        this.retryDelay = Math.min(this.retryDelay * 2, 8000);
      }
    };
    this.ws.onerror = () => this.ws?.close();
  }

  private send(message: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  appendEvent(event: Record<string, unknown>, actor: string): void {
    this.send({ type: "append", event, actor });
  }

  undo(actor: string): void {
    this.send({ type: "undo", actor });
  }

  close(): void {
    this.closedByUser = true;
    this.ws?.close();
  }
}
