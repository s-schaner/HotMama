from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from llm.parsers import RallySchema, parse_rally


def test_parse_clean_json():
    payload = {
        "rally_state": "serve_start",
        "who_won": "teamA",
        "reason": "serve_ace",
        "timeline": [{"t": 0.0, "event": "serve", "actor": "teamA#12"}],
    }
    result = parse_rally(str(payload).replace("'", '"'))
    assert isinstance(result, RallySchema)
    assert result.who_won == "teamA"


def test_parse_inner_json():
    messy = "Analysis: {\"rally_state\": \"rally\", \"who_won\": null, \"timeline\": []} Thanks!"
    result = parse_rally(messy)
    assert result.rally_state == "rally"


def test_parse_failure():
    with pytest.raises(ValueError):
        parse_rally("no json here")
