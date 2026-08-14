import { useState } from "react";

import { confirmProposal, dismissProposal } from "../api";
import type { MatchStateDto, ProposalDto } from "../types";

function describeProposal(proposal: ProposalDto, state: MatchStateDto): string {
  const event = proposal.proposal;
  const type = String(event.type ?? proposal.kind);
  if (type === "rally_ended") {
    const team = event.winner === "us" ? state.our_team : state.opponent;
    const reason = String(event.reason ?? "").replace("err_", "error: ");
    return `Rally ended — point ${team}${reason ? ` (${reason})` : ""}`;
  }
  if (type === "rally_started") return "Rally started";
  if (type === "moment_tagged") return `Tag: ${String(event.tag ?? "moment")}`;
  return `${proposal.kind} → ${type}`;
}

/** CV suggestions awaiting the human verdict — the trust gate (D-flow). */
export function ProposalsPanel({
  state,
  sessionId,
  actor,
  notify,
}: {
  state: MatchStateDto;
  sessionId: string;
  actor: string;
  notify: (text: string) => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  if (state.proposals.length === 0) return null;

  const act = async (
    proposal: ProposalDto,
    verdict: typeof confirmProposal | typeof dismissProposal,
    label: string,
  ) => {
    setBusy(proposal.event_id);
    try {
      await verdict(sessionId, proposal.event_id, actor);
      notify(label);
    } catch (err) {
      notify(`⚠ ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel">
      <h3>🤖 Needs your call ({state.proposals.length})</h3>
      <div className="join-list">
        {state.proposals.map((proposal) => (
          <div key={proposal.event_id} className="row" style={{ flexWrap: "nowrap" }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="label">{describeProposal(proposal, state)}</div>
              <div className="subtle">
                {Math.round(proposal.confidence * 100)}% · {proposal.producer}
                {proposal.actor ? ` · ${proposal.actor}` : ""}
              </div>
            </div>
            <button
              className="small primary"
              disabled={busy === proposal.event_id}
              onClick={() => act(proposal, confirmProposal, "Confirmed")}
            >
              ✓
            </button>
            <button
              className="small danger"
              disabled={busy === proposal.event_id}
              onClick={() => act(proposal, dismissProposal, "Dismissed")}
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
