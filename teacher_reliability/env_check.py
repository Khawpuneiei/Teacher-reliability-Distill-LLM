"""Check the local Python/CUDA/4-bit runtime before downloading a model."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from typing import Any


REQUIRED_PACKAGES = ("torch", "transformers", "accelerate", "bitsandbytes", "hf_xet")


def build_environment_report(
    *,
    python_version: tuple[int, int, int],
    cuda_available: bool,
    cuda_runtime: str | None,
    gpu_name: str | None,
    gpu_memory_bytes: int | None,
    package_versions: dict[str, str],
    bitsandbytes_import_error: str | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if python_version < (3, 10, 0):
        errors.append("Python 3.10 or newer is required.")
    if not cuda_available or not cuda_runtime:
        errors.append("CUDA-enabled PyTorch is unavailable; install the pinned CUDA wheel.")
    missing = [
        name for name in REQUIRED_PACKAGES
        if package_versions.get(name, "not-installed") == "not-installed"
    ]
    if missing:
        errors.append("Required packages are missing: " + ", ".join(missing) + ".")
    if bitsandbytes_import_error:
        errors.append("bitsandbytes could not load: " + bitsandbytes_import_error)
    if gpu_memory_bytes is not None and gpu_memory_bytes < 6 * 1024**3:
        warnings.append("GPU has less than 6 GiB VRAM; 4-bit model loading may be tight.")
    return {
        "ready": not errors,
        "python_version": ".".join(str(part) for part in python_version),
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_runtime,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "package_versions": package_versions,
        "errors": errors,
        "warnings": warnings,
    }


def collect_environment_report() -> dict[str, Any]:
    import torch

    package_versions = {}
    for package in REQUIRED_PACKAGES:
        distribution = "huggingface-hub" if package == "huggingface-hub" else package
        try:
            package_versions[package] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"

    bnb_import_error = None
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        bnb_import_error = f"{type(exc).__name__}: {exc}"

    gpu_name = None
    gpu_memory_bytes = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    return build_environment_report(
        python_version=tuple(sys.version_info[:3]),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_runtime=torch.version.cuda,
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory_bytes,
        package_versions=package_versions,
        bitsandbytes_import_error=bnb_import_error,
    )


def main() -> int:
    report = collect_environment_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
