from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType


def _samefile(path1: str, path2: str) -> bool:
    try:
        return os.path.samefile(path1, path2)
    except (FileNotFoundError, NotImplementedError, OSError, ValueError):
        return os.path.abspath(path1) == os.path.abspath(path2)


_real_gradio: ModuleType | None = None

if os.environ.get("GRADIO_STUB_ONLY") != "1":
    _stub_dir = os.path.dirname(__file__)
    _stub_root = os.path.dirname(_stub_dir)
    _checked_paths: list[str] = []

    for _entry in sys.path:
        _candidate = _entry or os.getcwd()
        if not _candidate or _candidate in _checked_paths:
            continue
        _checked_paths.append(_candidate)
        if _samefile(_candidate, _stub_root) or _samefile(_candidate, _stub_dir):
            continue

        _spec = importlib.machinery.PathFinder.find_spec(__name__, [_candidate])
        if not _spec or not _spec.loader or not _spec.origin:
            continue
        if _samefile(_spec.origin, __file__):
            continue

        _module = importlib.util.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_module)  # type: ignore[call-arg]
        except ModuleNotFoundError:
            continue
        _real_gradio = _module
        break

if _real_gradio is not None:
    sys.modules[__name__] = _real_gradio
    for _name, _value in _real_gradio.__dict__.items():
        if _name in {"__name__", "__loader__", "__spec__", "__package__", "__path__", "__file__"}:
            globals()[_name] = _value
            continue
        globals()[_name] = _value
else:
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
