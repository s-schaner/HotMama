"""Post-session report: the state + summary the dashboards already use,
rendered as a print-clean HTML document, optionally to PDF via WeasyPrint.

Input shapes are the wire dicts (``MatchState.to_public_dict()`` and
``match_summary``) — the report renders exactly what the coach saw live,
because it is derived from exactly the same event log.
"""

from __future__ import annotations

from html import escape
from typing import Any


class ReportUnavailableError(RuntimeError):
    """PDF rendering requested but WeasyPrint is not installed."""


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(value * 100)}%"


def _diff(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def _top_reason(lost_reasons: dict[str, int], labels: dict[str, str]) -> str:
    if not lost_reasons:
        return "—"
    reason, count = max(lost_reasons.items(), key=lambda item: item[1])
    return f"{labels.get(reason, reason)} ({count})"


def _set_rows(state: dict[str, Any]) -> str:
    rows = []
    for set_dto in state["sets"]:
        winner = set_dto.get("won_by")
        outcome = (
            escape(state["our_team"]) if winner == "us"
            else escape(state["opponent"]) if winner == "them"
            else "in progress"
        )
        rows.append(
            f"<tr><td>Set {set_dto['set_number']}</td>"
            f"<td class='num'>{set_dto['us_points']}–{set_dto['them_points']}</td>"
            f"<td>{outcome}</td>"
            f"<td class='num'>{set_dto['total_points']}</td></tr>"
        )
    return "".join(rows)


def _rotation_rows(summary: dict[str, Any]) -> str:
    labels = summary.get("loss_labels", {})
    rows = []
    for row in summary["rotation_table"]:
        rows.append(
            f"<tr><td>R{row['rotation']}</td>"
            f"<td class='num'>{row['serve_won']}–{row['serve_lost']}</td>"
            f"<td class='num'>{row['recv_won']}–{row['recv_lost']}</td>"
            f"<td class='num'>{_pct(row['side_out_pct'])}</td>"
            f"<td class='num'>{_diff(row['point_diff'])}</td>"
            f"<td>{escape(_top_reason(row['lost_reasons'], labels))}</td></tr>"
        )
    return "".join(rows)


def _player_rows(summary: dict[str, Any]) -> str:
    rows = []
    for line in summary["player_stats"]:
        played = (
            line["kills"] or line["aces"] or line["blocks"]
            or line["total_errors"] or line["serves"] or line["tags"]
        )
        if not played:
            continue
        jersey = f"#{line['jersey']}" if line["jersey"] is not None else ""
        tags = ", ".join(
            f"{key.replace('_', ' ')}×{count}" for key, count in sorted(line["tags"].items())
        )
        rows.append(
            f"<tr><td>{jersey} {escape(line['name'])}</td>"
            f"<td class='num'>{line['kills']}</td>"
            f"<td class='num'>{line['aces']}</td>"
            f"<td class='num'>{line['blocks']}</td>"
            f"<td class='num'>{line['total_errors']}</td>"
            f"<td class='num'>{line['serves']}</td>"
            f"<td>{escape(tags)}</td></tr>"
        )
    return "".join(rows) or "<tr><td colspan='7'>No attributed actions yet.</td></tr>"


def _runs_list(state: dict[str, Any], summary: dict[str, Any]) -> str:
    items = []
    for run in summary["scoring_runs"]:
        team = state["our_team"] if run["team"] == "us" else state["opponent"]
        items.append(
            f"<li>Set {run['set_number']}: {run['length']}-point run for "
            f"{escape(team)} ({run['from']} → {run['to']})</li>"
        )
    return "".join(items) or "<li>No runs of 3+ points.</li>"


def _moment_rows(state: dict[str, Any], clips: list[dict[str, Any]]) -> str:
    clip_by_event = {clip["event_id"]: clip for clip in clips if clip.get("event_id")}
    rows = []
    for tag in state["tags"]:
        clip = clip_by_event.get(tag["event_id"])
        clip_note = "✂ clip" if clip and clip.get("status") == "ready" else ""
        label = tag.get("custom_label") or tag["tag"].replace("_", " ")
        when = f"Set {tag['set_number']} · {tag['us_points']}–{tag['them_points']}"
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{when}</td>"
            f"<td>{escape(tag.get('note') or '')}</td><td>{clip_note}</td></tr>"
        )
    return "".join(rows) or "<tr><td colspan='4'>No tagged moments.</td></tr>"


_CSS = """
@page { size: A4; margin: 18mm 15mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       color: #16202a; margin: 0; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 20pt; margin: 0; }
h2 { font-size: 12pt; text-transform: uppercase; letter-spacing: 0.06em;
     color: #5a6b7a; border-bottom: 2px solid #e3e8ee; padding-bottom: 4px;
     margin: 22px 0 8px; }
.meta { color: #5a6b7a; margin-top: 2px; }
.final { font-size: 26pt; font-weight: 800; margin: 10px 0 2px; }
.final .us { color: #d95d2a; }
.final .them { color: #2a6bd9; }
table { width: 100%; border-collapse: collapse; margin-top: 6px; }
th { text-align: left; font-size: 9pt; text-transform: uppercase;
     letter-spacing: 0.05em; color: #5a6b7a; border-bottom: 1px solid #c9d2dc;
     padding: 4px 6px; }
td { padding: 5px 6px; border-bottom: 1px solid #edf1f5; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.callout { background: #fff4e5; border: 1px solid #f0c07f; border-radius: 6px;
           padding: 10px 12px; font-weight: 600; margin-top: 8px; }
ul { margin: 6px 0 0 18px; padding: 0; }
.footer { margin-top: 26px; color: #8896a5; font-size: 8pt; }
"""


def render_report_html(
    *,
    state: dict[str, Any],
    summary: dict[str, Any],
    clips: list[dict[str, Any]],
    label: str,
    generated_at: str,
) -> str:
    leak = summary.get("biggest_leak")
    leak_html = (
        f"<div class='callout'>⚠ {escape(leak['sentence'])}</div>" if leak else ""
    )
    our = escape(state["our_team"])
    their = escape(state["opponent"])
    title = escape(label) or f"{our} vs {their}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — HotMama report</title>
<style>{_CSS}</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">{escape(state['kind'])} · best of {state['best_of']} ·
    generated {escape(generated_at)}</div>
  <div class="final"><span class="us">{our} {state['sets_won_us']}</span>
    — <span class="them">{state['sets_won_them']} {their}</span></div>

  <h2>Sets</h2>
  <table>
    <tr><th>Set</th><th class="num">Score</th><th>Won by</th><th class="num">Rallies</th></tr>
    {_set_rows(state)}
  </table>

  <h2>Rotations</h2>
  {leak_html}
  <table>
    <tr><th>Rotation</th><th class="num">Serve W–L</th><th class="num">Receive W–L</th>
        <th class="num">Side-out %</th><th class="num">+/−</th><th>Top loss reason</th></tr>
    {_rotation_rows(summary)}
  </table>

  <h2>Players</h2>
  <table>
    <tr><th>Player</th><th class="num">K</th><th class="num">A</th><th class="num">B</th>
        <th class="num">Err</th><th class="num">Serves</th><th>Tags</th></tr>
    {_player_rows(summary)}
  </table>

  <h2>Momentum</h2>
  <ul>{_runs_list(state, summary)}</ul>

  <h2>Tagged moments</h2>
  <table>
    <tr><th>Moment</th><th>When</th><th>Note</th><th>Clip</th></tr>
    {_moment_rows(state, clips)}
  </table>

  <div class="footer">HotMama v2 — every number derives from the session event log
    ({state['applied_events']} events); corrections are replayed, never patched.</div>
</body>
</html>"""


def render_report_pdf(html: str) -> bytes:
    try:
        import weasyprint
    except ImportError as err:
        raise ReportUnavailableError(
            "PDF rendering needs WeasyPrint — install hotmama[reports]"
        ) from err
    return bytes(weasyprint.HTML(string=html).write_pdf())
