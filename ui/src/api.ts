// REST + WebSocket client. Same-origin in production (the host serves the PWA);
// the Vite dev server proxies /api and /ws to the host during development.

import type {
  CaptureStatusDto,
  ClipDto,
  ServerMessage,
  SessionListItem,
  SessionPayload,
} from "./types";

export async function listSessions(): Promise<SessionListItem[]> {
  const res = await fetch("/api/sessions");
  if (!res.ok) throw new Error(`list sessions failed: ${res.status}`);
  return res.json();
}

export async function getCaptureStatus(sessionId: string): Promise<CaptureStatusDto> {
  const res = await fetch(`/api/sessions/${sessionId}/capture`);
  if (!res.ok) throw new Error(`capture status failed: ${res.status}`);
  return res.json();
}

export async function startCapture(
  sessionId: string,
  source: string,
): Promise<CaptureStatusDto> {
  const res = await fetch(`/api/sessions/${sessionId}/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(body.detail ?? `start failed: ${res.status}`));
  }
  return res.json();
}

export async function stopCapture(sessionId: string): Promise<CaptureStatusDto> {
  const res = await fetch(`/api/sessions/${sessionId}/capture`, { method: "DELETE" });
  if (!res.ok) throw new Error(`stop failed: ${res.status}`);
  return res.json();
}

export async function getClips(sessionId: string): Promise<ClipDto[]> {
  const res = await fetch(`/api/sessions/${sessionId}/clips`);
  if (!res.ok) throw new Error(`clips failed: ${res.status}`);
  return res.json();
}

async function proposalVerdict(
  sessionId: string,
  observationId: string,
  verdict: "confirm" | "dismiss",
  actor: string,
): Promise<void> {
  const res = await fetch(
    `/api/sessions/${sessionId}/proposals/${observationId}/${verdict}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor }),
    },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(body.detail ?? `${verdict} failed: ${res.status}`));
  }
}

export const confirmProposal = (sessionId: string, id: string, actor: string) =>
  proposalVerdict(sessionId, id, "confirm", actor);
export const dismissProposal = (sessionId: string, id: string, actor: string) =>
  proposalVerdict(sessionId, id, "dismiss", actor);

export interface SavedRoster {
  name: string;
  players: {
    player_id: string;
    name: string;
    jersey: number | null;
    is_libero: boolean;
  }[];
  updated_at: string;
}

export async function listRosters(): Promise<SavedRoster[]> {
  const res = await fetch("/api/rosters");
  if (!res.ok) throw new Error(`rosters failed: ${res.status}`);
  return res.json();
}

export async function saveRoster(
  name: string,
  players: SavedRoster["players"],
): Promise<void> {
  const res = await fetch(`/api/rosters/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ players }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(body.detail ?? `save failed: ${res.status}`));
  }
}

export interface Venue {
  name: string;
  calibration: Record<string, unknown>;
  updated_at: string;
}

export async function listVenues(): Promise<Venue[]> {
  const res = await fetch("/api/venues");
  if (!res.ok) throw new Error(`venues failed: ${res.status}`);
  return res.json();
}

export async function saveVenue(
  name: string,
  calibration: Record<string, unknown>,
): Promise<void> {
  const res = await fetch(`/api/venues/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ calibration }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(body.detail ?? `save venue failed: ${res.status}`));
  }
}

export async function requestSummary(
  sessionId: string,
  actor: string,
): Promise<{ summary: string; model: string }> {
  const res = await fetch(`/api/sessions/${sessionId}/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actor }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(String(body.detail ?? `summary failed: ${res.status}`));
  }
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
  venue?: string | null;
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
