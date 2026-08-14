"""Between-set summary: compact facts in, three coachable paragraphs out.

The prompt feeds the LLM only numbers the engine already derived — the
model narrates, it never computes. If it can't be reached, the coach still
has every number on screen; summaries are strictly additive.
"""

from __future__ import annotations

import json
from typing import Any

from .llm import ChatClient

SYSTEM_PROMPT = """You are a volleyball assistant coach talking to the head coach
between sets. You are given verified match statistics as JSON. Write exactly three
short paragraphs, plain text, no headers or bullet points, 170 words maximum total:
1) what is working, 2) what is costing us points (be specific: rotation numbers and
reasons), 3) ONE concrete adjustment for the next set. Use the team names given.
Never invent numbers that are not in the data.""".strip()


def _condense_rotation(row: dict[str, Any]) -> dict[str, Any]:
    side_out = row.get("side_out_pct")
    return {
        "rotation": row.get("rotation"),
        "points": row.get("total"),
        "diff": row.get("point_diff"),
        "side_out_pct": round(side_out * 100) if side_out is not None else None,
        "lost_reasons": row.get("lost_reasons", {}),
    }


def build_summary_facts(state: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    players = [
        line
        for line in summary.get("player_stats", [])
        if line.get("kills") or line.get("aces") or line.get("blocks") or line.get("total_errors")
    ]
    current = state.get("current_set") or {}
    return {
        "our_team": state.get("our_team"),
        "opponent": state.get("opponent"),
        "sets": f"{state.get('sets_won_us')}-{state.get('sets_won_them')}",
        "current_set_score": (
            f"{current.get('us_points')}-{current.get('them_points')}"
            if current
            else None
        ),
        "set_scores": [
            f"{s.get('us_points')}-{s.get('them_points')}" for s in state.get("sets", [])
        ],
        "rotations": [
            _condense_rotation(row) for row in summary.get("rotation_table", [])
        ],
        "biggest_leak": (summary.get("biggest_leak") or {}).get("sentence"),
        "scoring_runs": summary.get("scoring_runs", []),
        "players": players,
        "loss_reason_labels": summary.get("loss_labels", {}),
    }


def generate_set_summary(
    client: ChatClient, state: dict[str, Any], summary: dict[str, Any]
) -> str:
    facts = build_summary_facts(state, summary)
    user = json.dumps(facts, separators=(",", ":"))
    return client.complete(SYSTEM_PROMPT, user)
