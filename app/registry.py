from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import Dict

from app.errors import PluginError
from plugins.plugins_api import AppContext, Plugin, SafePluginWrapper

LOGGER = logging.getLogger(__name__)


def discover_plugins(root: Path, ctx: AppContext) -> Dict[str, SafePluginWrapper]:
    plugins: Dict[str, SafePluginWrapper] = {}
    for plugin_dir in root.iterdir():
        if not plugin_dir.is_dir():
            continue
        module_path = plugin_dir / "plugin.py"
        if not module_path.exists():
            continue
        module_name = f"plugins.{plugin_dir.name}.plugin"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            LOGGER.exception("Failed to import plugin %s", module_name)
            continue
        plugin = _resolve_plugin(module)
        if plugin is None:
            continue
        wrapped = SafePluginWrapper(plugin, getattr(plugin, "name", plugin_dir.name))
        try:
            wrapped.on_register(ctx)
        except Exception:
            LOGGER.exception("Plugin %s failed during registration", wrapped.name)
            continue
        plugins[wrapped.name] = wrapped
    return plugins


def _resolve_plugin(module: ModuleType) -> Plugin | None:
    if hasattr(module, "PLUGIN"):
        plugin = getattr(module, "PLUGIN")
    elif hasattr(module, "get_plugin") and inspect.isfunction(module.get_plugin):
        plugin = module.get_plugin()
    else:
        LOGGER.warning("Module %s does not expose a plugin", module.__name__)
        return None

    if not hasattr(plugin, "on_register"):
        raise PluginError(f"Plugin {plugin} missing required interface")
    return plugin
