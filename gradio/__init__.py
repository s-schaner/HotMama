from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
import warnings


def _samefile(path1: str, path2: str) -> bool:
    try:
        return os.path.samefile(path1, path2)
    except (FileNotFoundError, NotImplementedError, OSError, ValueError):
        return os.path.abspath(path1) == os.path.abspath(path2)


_real_gradio: ModuleType | None = None

if os.environ.get("GRADIO_STUB_ONLY") != "1":
    _stub_dir = os.path.dirname(__file__)
    _stub_root = os.path.dirname(_stub_dir)

    def _is_stub_path(entry: str) -> bool:
        candidate = entry or os.getcwd()
        return _samefile(candidate, _stub_root) or _samefile(candidate, _stub_dir)

    _original_module = sys.modules.get(__name__)
    _original_sys_path = list(sys.path)
    try:
        sys.modules.pop(__name__, None)
        sys.path = [entry for entry in _original_sys_path if not _is_stub_path(entry)]
        _real_gradio = importlib.import_module(__name__)
    except ModuleNotFoundError:
        _real_gradio = None
    finally:
        sys.path = _original_sys_path
        if _real_gradio is None and _original_module is not None:
            sys.modules[__name__] = _original_module

if _real_gradio is not None:
    sys.modules[__name__] = _real_gradio
    for _name, _value in _real_gradio.__dict__.items():
        if _name in {
            "__name__",
            "__loader__",
            "__spec__",
            "__package__",
            "__path__",
            "__file__",
        }:
            globals()[_name] = _value
            continue
        globals()[_name] = _value
else:
    warnings.warn(
        "Real 'gradio' package not found. Using stub implementation; the UI will not launch."
        " Install the 'gradio' package or set GRADIO_STUB_ONLY=1 to silence this warning.",
        RuntimeWarning,
        stacklevel=2,
    )
    from typing import Any, Callable, List

    class Component:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def click(
            self,
            fn: Callable,
            inputs: List[Any] | None = None,
            outputs: Any | None = None,
        ) -> None:
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
