from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class HardwareReport:
    """Summary of detected graphics hardware and environment details."""

    vendor: str = "cpu"
    adapters: List[str] = field(default_factory=list)
    driver_info: List[str] = field(default_factory=list)
    wsl_version: str | None = None
    diagnostics: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = [f"Vendor: {self.vendor}"]
        if self.adapters:
            lines.append("Adapters: " + ", ".join(self.adapters))
        if self.driver_info:
            lines.append("Drivers: " + ", ".join(self.driver_info))
        if self.wsl_version:
            lines.append(f"WSL: {self.wsl_version}")
        for warning in self.warnings:
            lines.append(f"Warning: {warning}")
        return lines


def _run_command(cmd: Sequence[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)


def _detect_wsl() -> str | None:
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        release = ""
    release_upper = release.upper()
    if "WSL2" in release_upper:
        return "WSL2"
    if "WSL" in release_upper:
        return "WSL"
    if os.environ.get("WSL_INTEROP"):
        return "WSL"
    return None


def detect_graphics_stack() -> HardwareReport:
    report = HardwareReport()
    report.wsl_version = _detect_wsl()

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        report.vendor = "nvidia"
        code, out, err = _run_command(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        )
        if code == 0 and out:
            for line in out.splitlines():
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                if parts:
                    report.adapters.append(parts[0])
                if len(parts) > 1:
                    report.driver_info.append(f"driver {parts[1]}")
        else:
            if err:
                report.warnings.append(f"nvidia-smi diagnostics: {err}")
        return report

    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        report.vendor = "amd"
        code, out, err = _run_command([rocm_smi, "--showproductname", "--showdriverversion"])
        if code == 0 and out:
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "Card series" in line or "GPU" in line:
                    report.adapters.append(line.split(":", 1)[-1].strip())
                if "Driver version" in line:
                    report.driver_info.append(line.split(":", 1)[-1].strip())
        else:
            if err:
                report.warnings.append(f"rocm-smi diagnostics: {err}")
        return report

    lspci = shutil.which("lspci")
    if lspci:
        code, out, _ = _run_command([lspci])
        if code == 0 and out:
            for line in out.splitlines():
                if "VGA" in line or "3D controller" in line:
                    clean = line.split(":", 2)
                    adapter = clean[-1].strip() if clean else line.strip()
                    report.adapters.append(adapter)
                    if "NVIDIA" in adapter.upper():
                        report.vendor = "nvidia"
                    elif "AMD" in adapter.upper() or "RADEON" in adapter.upper():
                        report.vendor = "amd"
                    elif "INTEL" in adapter.upper():
                        report.vendor = "intel"
    if not report.adapters:
        report.warnings.append("No discrete GPU detected; defaulting to CPU profile")
    if report.vendor not in {"nvidia", "amd", "intel"}:
        report.vendor = "cpu"
    return report


def _format_version(value: Tuple[int, int] | Tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in value)


@dataclass(frozen=True)
class DependencyProfile:
    key: str
    description: str
    python_min: Tuple[int, int]
    python_max: Tuple[int, int] | None
    core_packages: List[str]
    opencv_candidates: List[str]
    gpu_packages: List[Sequence[str]]
    extras: List[Sequence[str]]
    import_checks: dict[str, Tuple[Tuple[int, int], Tuple[int, int] | None]]
    flash_attn: Sequence[str] | None = None

    def python_requirement(self) -> str:
        minimum = _format_version(self.python_min)
        maximum = _format_version(self.python_max) if self.python_max else None
        if maximum:
            return f">= {minimum}, <= {maximum}"
        return f">= {minimum}"


CORE_PACKAGES_PINNED = [
    "fastapi==0.110.0",
    "uvicorn[standard]==0.29.0",
    "jinja2==3.1.4",
    "python-multipart==0.0.9",
    "requests==2.31.0",
    "pillow==10.3.0",
    "numpy==1.26.4",
    "pydantic==1.10.15",
    "ultralytics==8.2.55",
    "matplotlib==3.8.4",
]

OPENCV_PINNED = [
    "opencv-python-headless==4.9.0.80",
    "opencv-python==4.9.0.80",
]

IMPORT_CHECKS_DEFAULT = {
    "fastapi": ((3, 10), None),
    "numpy": ((3, 10), None),
    "matplotlib": ((3, 10), None),
}


PROFILES: dict[str, DependencyProfile] = {
    "cpu": DependencyProfile(
        key="cpu",
        description="CPU / generic profile",
        python_min=(3, 10),
        python_max=(3, 12),
        core_packages=CORE_PACKAGES_PINNED,
        opencv_candidates=OPENCV_PINNED,
        gpu_packages=[
            ["torch==2.3.1", "torchvision==0.18.1"],
        ],
        extras=[["onnxruntime==1.18.0"]],
        import_checks=IMPORT_CHECKS_DEFAULT,
        flash_attn=None,
    ),
    "nvidia": DependencyProfile(
        key="nvidia",
        description="NVIDIA CUDA profile",
        python_min=(3, 10),
        python_max=(3, 12),
        core_packages=CORE_PACKAGES_PINNED,
        opencv_candidates=OPENCV_PINNED,
        gpu_packages=[
            [
                "torch==2.3.1+cu121",
                "torchvision==0.18.1+cu121",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        ],
        extras=[["onnxruntime-gpu==1.18.0"]],
        import_checks=IMPORT_CHECKS_DEFAULT,
        flash_attn=["flash-attn==2.5.9", "--no-build-isolation"],
    ),
    "amd": DependencyProfile(
        key="amd",
        description="AMD ROCm or DirectML profile",
        python_min=(3, 10),
        python_max=(3, 12),
        core_packages=CORE_PACKAGES_PINNED,
        opencv_candidates=OPENCV_PINNED,
        gpu_packages=[["torch==2.3.1", "torchvision==0.18.1"]],
        extras=[["onnxruntime==1.18.0"]],
        import_checks=IMPORT_CHECKS_DEFAULT,
        flash_attn=None,
    ),
}


def resolve_profile(preferred: str | None = None, hardware: HardwareReport | None = None) -> tuple[DependencyProfile, HardwareReport]:
    hardware = hardware or detect_graphics_stack()
    key = (preferred or "auto").lower()
    if key in {"auto", "", "default"}:
        vendor_key = hardware.vendor.lower()
        key = vendor_key if vendor_key in PROFILES else "cpu"
    if key not in PROFILES:
        key = "cpu"
    return PROFILES[key], hardware


def python_version_supported(profile: DependencyProfile, version_info: tuple[int, int, int]) -> bool:
    minimum = profile.python_min
    maximum = profile.python_max
    if version_info[: len(minimum)] < minimum:
        return False
    if maximum and version_info[: len(maximum)] > maximum:
        return False
    return True


def format_environment_summary(profile: DependencyProfile, hardware: HardwareReport) -> List[str]:
    lines = [f"Dependency profile: {profile.description} ({profile.key})"]
    lines.extend(hardware.summary_lines())
    lines.append(f"Python requirement: {profile.python_requirement()}")
    lines.append(f"Python runtime: {platform.python_version()}")
    return lines
