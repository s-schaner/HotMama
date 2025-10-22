"""Configuration manager for saving and loading user LLM settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("hotmama.gui.config_manager")


class ConfigManager:
    """Manages saving and loading of user LLM configurations."""

    def __init__(self, config_dir: Path) -> None:
        """Initialize the config manager.

        Args:
            config_dir: Directory to store configuration files
        """
        self._config_dir = config_dir
        self._config_file = config_dir / "llm_configs.json"
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, name: str, config: dict[str, Any]) -> None:
        """Save a configuration with a given name.

        Args:
            name: Unique name for this configuration
            config: Configuration dictionary containing endpoint, key, model, etc.
        """
        try:
            configs = self._load_all_configs()
            configs[name] = config

            with open(self._config_file, "w") as f:
                json.dump(configs, f, indent=2)

            LOGGER.info(f"Saved configuration '{name}'")
        except Exception as exc:
            LOGGER.error(f"Failed to save configuration '{name}'", exc_info=exc)
            raise

    def load_config(self, name: str) -> dict[str, Any] | None:
        """Load a configuration by name.

        Args:
            name: Name of the configuration to load

        Returns:
            Configuration dictionary or None if not found
        """
        try:
            configs = self._load_all_configs()
            return configs.get(name)
        except Exception as exc:
            LOGGER.error(f"Failed to load configuration '{name}'", exc_info=exc)
            return None

    def list_configs(self) -> list[str]:
        """List all saved configuration names.

        Returns:
            List of configuration names
        """
        try:
            configs = self._load_all_configs()
            return list(configs.keys())
        except Exception:
            return []

    def delete_config(self, name: str) -> None:
        """Delete a configuration by name.

        Args:
            name: Name of the configuration to delete
        """
        try:
            configs = self._load_all_configs()
            if name in configs:
                del configs[name]
                with open(self._config_file, "w") as f:
                    json.dump(configs, f, indent=2)
                LOGGER.info(f"Deleted configuration '{name}'")
        except Exception as exc:
            LOGGER.error(f"Failed to delete configuration '{name}'", exc_info=exc)
            raise

    def _load_all_configs(self) -> dict[str, Any]:
        """Load all configurations from file.

        Returns:
            Dictionary of all configurations
        """
        if not self._config_file.exists():
            return {}

        try:
            with open(self._config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            LOGGER.warning("Config file corrupted, returning empty configs")
            return {}
        except Exception as exc:
            LOGGER.error("Failed to load configs", exc_info=exc)
            return {}


__all__ = ["ConfigManager"]
