from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, List


class Component:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def click(self, fn: Callable, inputs: List[Any] | None = None, outputs: Any | None = None) -> None:
        self._callback = fn
        self._inputs = inputs
        self._outputs = outputs


class Tab(Component):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class Blocks(Component):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def launch(self, **kwargs: Any) -> None:
        return None


class Textbox(Component):
    pass


class Number(Component):
    pass


class Button(Component):
    pass


class JSON(Component):
    pass


class Markdown(Component):
    pass


class Image(Component):
    pass


class Radio(Component):
    pass
