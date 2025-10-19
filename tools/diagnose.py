from __future__ import annotations

import json
import platform
import shutil
from dataclasses import asdict, dataclass


@dataclass
class EnvReport:
    python: str
    platform: str
    gpu_available: bool
    cuda_toolkit: bool


def detect_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def detect_cuda() -> bool:
    return shutil.which("nvcc") is not None


def main() -> None:
    report = EnvReport(
        python=platform.python_version(),
        platform=platform.platform(),
        gpu_available=detect_gpu(),
        cuda_toolkit=detect_cuda(),
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
