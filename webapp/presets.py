import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass
class EndpointPreset:
    name: str
    endpoint: str
    model: str
    token: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


DEFAULT_PRESETS: List[EndpointPreset] = [
    EndpointPreset(
        name="Hugging Face Qwen 32B",
        endpoint="https://xw3xbts6l8qsa6x9.us-east-2.aws.endpoints.huggingface.cloud",
        model="Qwen/Qwen2.5-VL-32B-Instruct",
        token=(
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_API_TOKEN")
            or "hf_your_api_token"
        ),
    ),
    EndpointPreset(
        name="Local LM Studio Qwen 7B",
        endpoint="http://192.168.86.29:1234",
        model="qwen/qwen2.5-vl-7b",
        token="xxx",
    ),
]


def _read_presets(path: Path) -> List[EndpointPreset]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        # Corrupt file; start from defaults but preserve backup
        try:
            backup = path.with_suffix(".bak")
            path.rename(backup)
        except OSError:
            pass
        return []
    presets: List[EndpointPreset] = []
    for item in payload if isinstance(payload, list) else []:
        try:
            presets.append(
                EndpointPreset(
                    name=str(item.get("name", "Unnamed")),
                    endpoint=str(item.get("endpoint", "")),
                    token=str(item.get("token", "")),
                    model=str(item.get("model", "")),
                )
            )
        except Exception:
            continue
    return presets


def _write_presets(path: Path, presets: Iterable[EndpointPreset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [preset.to_dict() for preset in presets]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_presets(path: Path) -> List[EndpointPreset]:
    presets = _read_presets(path)
    if not presets:
        presets = DEFAULT_PRESETS.copy()
        _write_presets(path, presets)
    return presets


def upsert_preset(path: Path, preset: EndpointPreset) -> EndpointPreset:
    presets = load_presets(path)
    existing = {p.name: p for p in presets}
    existing[preset.name] = preset
    ordered = sorted(existing.values(), key=lambda p: p.name.lower())
    _write_presets(path, ordered)
    return preset


def save_presets(path: Path, presets: Iterable[EndpointPreset]) -> None:
    ordered = sorted(presets, key=lambda p: p.name.lower())
    _write_presets(path, ordered)
