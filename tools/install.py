from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tools.dependencies import (
    format_environment_summary,
    python_version_supported,
    resolve_profile,
)


def torch_env() -> tuple[str, str, str]:
    try:
        import torch  # type: ignore[import-not-found]

        cuda = torch.version.cuda or ""
        sm = ""
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
        return torch.__version__, cuda, sm
    except Exception:
        return "", "", ""


def flash_attn_supported(torch_ver: str, cuda_ver: str, sm: str) -> bool:
    """
    Be conservative: only allow when wheels are typically available.
    Adjust as upstream support grows.
    """

    if not torch_ver or not cuda_ver:
        return False
    if not (
        cuda_ver.startswith(
            (
                "11.8",
                "12.1",
                "12.2",
                "12.3",
                "12.4",
                "12.5",
                "12.6",
                "12.7",
            )
        )
    ):
        return False
    if sm in ("sm_80", "sm_86", "sm_89", "sm_90"):
        return True
    return False

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


def run_pip(args: Sequence[str]) -> subprocess.CompletedProcess:
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


def attempt_install(package: Sequence[str] | str, report: InstallReport) -> bool:
    if isinstance(package, str):
        args = [package]
    else:
        args = list(package)
    result = run_pip(args)
    success = result.returncode == 0
    message_source = result.stdout if success else result.stderr or result.stdout
    message_lines = (message_source or "").strip().splitlines()
    message = message_lines[-1] if message_lines else ""
    report.record(args[0], success, message)
    status = "succeeded" if success else "failed"
    _log(f"Result: {args[0]} {status}")
    return success


def _format_py_requirement(min_version: tuple[int, int], max_version: tuple[int, int] | None) -> str:
    minimum = ".".join(str(v) for v in min_version)
    if not max_version:
        return f">= {minimum}"
    maximum = ".".join(str(v) for v in max_version)
    return f">= {minimum}, <= {maximum}"


def check_imports(
    packages: List[str],
    report: InstallReport,
    python_requirements: Dict[str, tuple[tuple[int, int], tuple[int, int] | None]],
) -> None:
    _log("\nVerifying imports...")
    for pkg in packages:
        min_py, max_py = python_requirements.get(pkg, ((3, 10), None))
        current_version = sys.version_info[: len(min_py)]
        max_slice = sys.version_info[: len(max_py)] if max_py else None
        if current_version < min_py or (max_py and max_slice and max_slice > max_py):
            requirement_str = _format_py_requirement(min_py, max_py)
            msg = f"requires Python {requirement_str}, current {platform.python_version()}"
            report.record(pkg, False, msg)
            _log(f"❌ {pkg}: {msg}")
            continue
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


def finalize(report: InstallReport, exit_code: int) -> int:
    report.print()
    _log(f"Log saved to: {LOG_PATH}")
    return exit_code


def maybe_adjust_drivers(hardware, attempt_adjust: bool) -> None:
    _log("\nReviewing GPU driver state...")
    if not attempt_adjust:
        _log("Driver adjustments skipped (pass --attempt-driver-adjust to enable).")
        return

    if hardware.vendor == "nvidia" and shutil.which("nvidia-smi"):
        _log("Attempting to enable NVIDIA persistence mode for stability...")
        result = subprocess.run(
            ["nvidia-smi", "-pm", "1"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _log("✅ NVIDIA persistence mode enabled or already active.")
        else:
            _log("❌ NVIDIA driver command failed; see details below.")
            diagnostics = (result.stderr or result.stdout or "").strip().splitlines()
            _log_block("  diagnostics:", diagnostics[-5:])
        return

    if hardware.vendor == "amd" and shutil.which("rocm-smi"):
        _log("Attempting to query AMD ROCm driver status...")
        result = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = (result.stdout or "").strip().splitlines()
            _log_block("  driver info:", lines[-5:])
            _log("✅ AMD ROCm diagnostics completed.")
        else:
            _log("❌ AMD ROCm diagnostics failed; see details below.")
            diagnostics = (result.stderr or result.stdout or "").strip().splitlines()
            _log_block("  diagnostics:", diagnostics[-5:])
        return

    _log("No GPU driver utilities detected; nothing to adjust.")


def install(profile_key: str = "auto", attempt_driver_adjust: bool = False) -> int:
    _init_log()
    report = InstallReport()

    profile, hardware = resolve_profile(profile_key)
    _log("Resolved dependency profile and environment:")
    for line in format_environment_summary(profile, hardware):
        _log(f"  {line}")

    maybe_adjust_drivers(hardware, attempt_driver_adjust)

    if not python_version_supported(profile, sys.version_info[:3]):
        requirement = profile.python_requirement()
        message = f"Python {platform.python_version()} outside supported range ({requirement})"
        _log(f"❌ {message}")
        report.record("python", False, message)
        return finalize(report, 1)

    _log("\nInstalling core packages...")
    for pkg in profile.core_packages:
        if not attempt_install(pkg, report):
            _log(f"Failed to install core package {pkg}; aborting early.")
            return finalize(report, 1)

    _log("Installing OpenCV with fallback...")
    for pkg in profile.opencv_candidates:
        if attempt_install(pkg, report):
            break
    else:
        _log("OpenCV packages failed; continuing without them")

    if profile.gpu_packages:
        _log("Installing GPU/runtime packages...")
        for pkg in profile.gpu_packages:
            attempt_install(pkg, report)

    if profile.extras:
        _log("Installing optional extras...")
        for pkg in profile.extras:
            attempt_install(pkg, report)

    tver, cver, sm = torch_env()
    if profile.flash_attn:
        if flash_attn_supported(tver, cver, sm):
            attempt_install(profile.flash_attn, report)
        else:
            message = f"skipped (torch={tver}, cuda={cver}, arch={sm})"
            report.record(profile.flash_attn[0], False, message)
            _log(f"❌ {profile.flash_attn[0]}: {message}")
    else:
        _log("flash-attn not requested for this profile; skipping.")

    check_imports(list(profile.import_checks.keys()), report, profile.import_checks)

    return finalize(report, 0 if not report.failure else 1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VolleySense dependency installer")
    parser.add_argument(
        "--profile",
        choices=["auto", "cpu", "nvidia", "amd"],
        default="auto",
        help="Select dependency profile (default: auto-detect)",
    )
    parser.add_argument(
        "--attempt-driver-adjust",
        action="store_true",
        help="Attempt to adjust GPU driver state when supported",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    sys.exit(install(cli_args.profile, cli_args.attempt_driver_adjust))
