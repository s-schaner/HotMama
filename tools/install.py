from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

CORE_PACKAGES = ["gradio", "requests", "pillow", "numpy", "ultralytics", "matplotlib"]
OPENCV_PACKAGES = ["opencv-python-headless", "opencv-python"]
GPU_PACKAGES = [
    ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
]
EXTRAS = [["onnxruntime-gpu"], ["flash-attn"], ["tensorrt"]]
LOG_PATH = Path(__file__).with_name("install.log")


def _init_log() -> None:
    LOG_PATH.write_text(
        "VolleySense installer run\n"
        f"Started at: {datetime.utcnow().isoformat()}Z\n"
        f"Python: {sys.version.splitlines()[0]}\n"
        f"Platform: {platform.platform()}\n"
        f"Executable: {sys.executable}\n\n"
    )


def _log(message: str) -> None:
    print(message)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")


def _log_block(header: str, lines: Iterable[str]) -> None:
    if not lines:
        return
    _log(header)
    for line in lines:
        _log(f"    {line}")


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
        _log("\nInstallation summary:")
        for pkg, msg in self.success.items():
            _log(f"✅ {pkg}: {msg}")
        for pkg, msg in self.failure.items():
            _log(f"❌ {pkg}: {msg}")
        if self.failure:
            _log("\nSome packages failed to install. See install.log for full diagnostics.")


def run_pip(args: List[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "pip", "install", *args]
    _log(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    stdout_lines = (result.stdout or "").strip().splitlines()
    stderr_lines = (result.stderr or "").strip().splitlines()
    tail = stdout_lines[-5:] if stdout_lines else []
    err_tail = stderr_lines[-5:] if stderr_lines else []
    if tail:
        _log_block("  stdout tail:", tail)
    if err_tail:
        _log_block("  stderr tail:", err_tail)
    _log(f"  return code: {result.returncode}")
    return result


def attempt_install(package: List[str] | str, report: InstallReport) -> bool:
    args = package if isinstance(package, list) else [package]
    result = run_pip(args)
    success = result.returncode == 0
    message_source = result.stdout if success else result.stderr or result.stdout
    message = (message_source.strip().splitlines() or [""])[-1]
    report.record(args[0], success, message)
    status = "succeeded" if success else "failed"
    _log(f"Result: {args[0]} {status}")
    return success


def check_imports(packages: List[str], report: InstallReport) -> None:
    _log("\nVerifying imports...")
    for pkg in packages:
        try:
            __import__(pkg)
        except Exception as exc:  # noqa: BLE001 - we want full diagnostics
            msg = f"import failed: {exc}"
            report.record(pkg, False, msg)
            _log(f"❌ Unable to import {pkg}: {exc}")
        else:
            if pkg not in report.success:
                report.record(pkg, True, "imported (pre-installed)")
            _log(f"✅ Imported {pkg}")


def install() -> int:
    _init_log()
    report = InstallReport()
    _log("Installing core packages...")
    for pkg in CORE_PACKAGES:
        if not attempt_install(pkg, report):
            _log(f"Failed to install core package {pkg}; aborting early.")
            return finalize(report, 1)

    _log("Installing OpenCV with fallback...")
    for pkg in OPENCV_PACKAGES:
        if attempt_install(pkg, report):
            break
    else:
        _log("OpenCV packages failed; continuing without them")

    _log("Attempting GPU packages...")
    for pkg in GPU_PACKAGES:
        attempt_install(pkg, report)

    _log("Installing optional extras...")
    for pkg in EXTRAS:
        attempt_install(pkg, report)

    check_imports(["gradio", "numpy", "matplotlib"], report)

    return finalize(report, 0 if not report.failure else 1)


def finalize(report: InstallReport, exit_code: int) -> int:
    report.print()
    _log(f"Log saved to: {LOG_PATH}")
    return exit_code


if __name__ == "__main__":
    sys.exit(install())
