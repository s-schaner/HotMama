from __future__ import annotations

from pathlib import Path

from app.registry import discover_plugins
from plugins.plugins_api import AppContext


def test_discover_plugins(tmp_path: Path) -> None:
    plugins_dir = Path("plugins")
    ctx = AppContext(db_url="sqlite:///:memory:", sessions_dir=tmp_path, config={})
    plugins = discover_plugins(plugins_dir, ctx)
    assert "pose_per_player" in plugins
    assert all(hasattr(plugin, "on_register") for plugin in plugins.values())
