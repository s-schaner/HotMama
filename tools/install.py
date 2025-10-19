from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List

CORE_PACKAGES = ["gradio", "requests", "pillow", "numpy", "ultralytics", "matplotlib"]
OPENCV_PACKAGES = ["opencv-python-headless", "opencv-python"]
GPU_PACKAGES = [
    ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
]
EXTRAS = [["onnxruntime-gpu"], ["flash-attn"], ["tensorrt"]]


@dataclass
class InstallReport:
    success: Dict[str, str] = field(default_factory=dict)
    failure: Dict[str, str] = field(default_factory=dict)

    def record(self, package: str, status: bool, message: str) -> None:
        if status:
            self.success[package] = message
        else:
            self.failure[package] = message

    def print(self) -> None:
        print("\nInstallation summary:")
        for pkg, msg in self.success.items():
            print(f"✅ {pkg}: {msg}")
        for pkg, msg in self.failure.items():
            print(f"❌ {pkg}: {msg}")
        if self.failure:
            print("\nSome packages failed to install. Consider manual installation.")


def run_pip(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", "install", *args], capture_output=True, text=True)


def attempt_install(package: List[str] | str, report: InstallReport) -> bool:
    args = package if isinstance(package, list) else [package]
    result = run_pip(args)
    success = result.returncode == 0
    message = result.stdout.strip() or result.stderr.strip()
    report.record(args[0], success, message.splitlines()[-1] if message else "")
    return success


def install() -> int:
    report = InstallReport()
    print("Installing core packages...")
    for pkg in CORE_PACKAGES:
        if not attempt_install(pkg, report):
            print(f"Failed to install core package {pkg}")
            return finalize(report, 1)

    print("Installing OpenCV with fallback...")
    for pkg in OPENCV_PACKAGES:
        if attempt_install(pkg, report):
            break
    else:
        print("OpenCV packages failed; continuing without them")

    print("Attempting GPU packages...")
    for pkg in GPU_PACKAGES:
        attempt_install(pkg, report)

    print("Installing optional extras...")
    for pkg in EXTRAS:
        attempt_install(pkg, report)

    return finalize(report, 0)


def finalize(report: InstallReport, exit_code: int) -> int:
    report.print()
    return exit_code


if __name__ == "__main__":
    sys.exit(install())
