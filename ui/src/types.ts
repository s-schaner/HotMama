// Wire types mirroring the server's state.to_public_dict() and match_summary().
// The UI is a renderer: it never re-derives volleyball rules.

export type Team = "us" | "them";

export type PointReason =
  | "kill"
  | "ace"
  | "block"
  | "err_serve"
  | "err_attack"
  | "err_net"
  | "err_handling"
  | "err_other"
  | "unknown";

export type TagCode =
  | "great_serve"
  | "great_dig"
  | "attack_winner"
  | "serve_error"
  | "dig_failure"
  | "positioning_error"
  | "highlight"
  | "custom";

export interface RosterPlayer {
  player_id: string;
  name: string;
  jersey: number | null;
  is_libero: boolean;
}

export interface RotationRow {
  rotation?: number;
  serve_won: number;
  serve_lost: number;
  recv_won: number;
  recv_lost: number;
  total: number;
  point_diff: number;
  side_out_pct: number | null;
  serve_win_pct: number | null;
  lost_reasons: Record<string, number>;
  won_reasons: Record<string, number>;
}

export interface PointDto {
  number: number;
  winner: Team;
  reason: PointReason;
  served_by: Team;
  our_rotation: number;
  our_server: string | null;
  player_id: string | null;
  opponent_jersey: number | null;
  us_points: number;
  them_points: number;
  event_id: string;
}

export interface SetDto {
  set_number: number;
  to_win: number;
  us_points: number;
  them_points: number;
  serving: Team;
  our_rotation: number;
  our_server: string;
  slot_owner: string[];
  on_court: string[];
  liberos: string[];
  finished: boolean;
  won_by: Team | null;
  decided: Team | null;
  set_point: Team | null;
  rally_in_progress: boolean;
  subs_used: number;
  timeouts: Record<string, number>;
  per_rotation: Record<string, RotationRow>;
  points: PointDto[];
  total_points: number;
}

export interface TagDto {
  event_id: string;
  tag: TagCode;
  player_id: string | null;
  note: string | null;
  custom_label: string | null;
  set_number: number;
  us_points: number;
  them_points: number;
  occurred_at: string;
}

export interface ProposalDto {
  event_id: string;
  kind: string;
  proposal: Record<string, unknown>;
  confidence: number;
  producer: string;
  actor: string | null;
  occurred_at: string;
}

export interface MatchStateDto {
  created: boolean;
  kind: string;
  our_team: string;
  opponent: string;
  best_of: number;
  set_points: number;
  final_set_points: number;
  roster: RosterPlayer[];
  sets_won_us: number;
  sets_won_them: number;
  match_over: boolean;
  session_closed: boolean;
  current_set: SetDto | null;
  sets: SetDto[];
  tags: TagDto[];
  notes: unknown[];
  warnings: string[];
  cv_observations: number;
  proposals: ProposalDto[];
  applied_events: number;
  last_event_id: string | null;
}

export interface LeakDto {
  rotation: number;
  reason: string;
  count: number;
  rotation_points_lost: number;
  sentence: string;
}

export interface PlayerLineDto {
  player_id: string;
  name: string;
  jersey: number | null;
  kills: number;
  aces: number;
  blocks: number;
  errors: Record<string, number>;
  total_errors: number;
  serves: number;
  tags: Record<string, number>;
}

export interface SummaryDto {
  rotation_table: RotationRow[];
  biggest_leak: LeakDto | null;
  player_stats: PlayerLineDto[];
  scoring_runs: {
    set_number: number;
    team: Team;
    length: number;
    from: string;
    to: string;
  }[];
  reason_labels: Record<string, string>;
  loss_labels: Record<string, string>;
}

export interface SessionPayload {
  type: "snapshot" | "event";
  last_seq: number;
  state: MatchStateDto;
  summary: SummaryDto;
  event?: Record<string, unknown>;
  seq?: number;
  retracted_event_id?: string;
}

export interface ServerError {
  type: "error";
  detail: string;
  errors?: unknown[];
}

export interface CaptureStatusDto {
  state: "idle" | "recording" | "finished" | "error";
  error?: string | null;
  source?: string;
  codec?: string;
  fps?: number;
  width?: number;
  height?: number;
  started_at?: string | null;
  segments?: number;
  frames_total?: number;
}

export interface ClipDto {
  clip_id: string;
  session_id: string;
  event_id: string;
  kind: "tag" | "rally";
  label: string;
  status: "pending" | "ready" | "failed";
  url: string | null;
  start_at: string;
  end_at: string;
  error: string | null;
  created_at: string;
}

export type ServerMessage =
  | SessionPayload
  | ServerError
  | { type: "pong" }
  | { type: "capture"; status: CaptureStatusDto }
  | { type: "clips"; clips: ClipDto[] };

export interface SessionListItem {
  session_id: string;
  created_at: string;
  kind: string;
  label: string;
  closed: number;
  last_seq: number;
}

export type Role = "coach" | "statter";
