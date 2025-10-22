"""Base class for GUI modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr

    from deploy.gui.app.controller import GuiController


class GuiModule(ABC):
    """Abstract base class for modular GUI components."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this module."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Return the human-readable name shown in the UI."""
        pass

    @abstractmethod
    def build_ui(self, controller: GuiController) -> gr.Blocks:
        """Build and return the Gradio UI components for this module.

        Args:
            controller: The GUI controller instance for business logic

        Returns:
            A Gradio Blocks instance containing the module's UI
        """
        pass


__all__ = ["GuiModule"]
