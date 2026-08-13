"""Report rendering from the shared analytics scenario."""

from __future__ import annotations

import pytest

from hotmama.analytics import match_summary
from hotmama.core import replay
from hotmama.reports import render_report_html
from tests.analytics.test_projections import scenario


def _html() -> str:
    state = replay(scenario())
    return render_report_html(
        state=state.to_public_dict(),
        summary=match_summary(state),
        clips=[],
        label="HotMama vs Rivals",
        generated_at="2026-08-13 18:00 UTC",
    )


def test_report_contains_core_sections() -> None:
    html = _html()
    assert "HotMama vs Rivals" in html
    assert "Rotations" in html
    assert "R3" in html
    assert "attack errors" in html  # the engineered leak
    assert "Rotation 3 is your biggest leak" in html
    assert "Player 4" in html  # top scorer
    assert "great dig×2" in html  # tag rollup for p6
    assert "Tagged moments" in html


def test_report_escapes_user_strings() -> None:
    state = replay(scenario())
    public = state.to_public_dict()
    public["our_team"] = "<script>alert(1)</script>"
    html = render_report_html(
        state=public,
        summary=match_summary(state),
        clips=[],
        label="",
        generated_at="now",
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_pdf_renders_when_weasyprint_present() -> None:
    pytest.importorskip("weasyprint")
    from hotmama.reports import render_report_pdf

    pdf = render_report_pdf(_html())
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 5000
