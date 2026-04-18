"""
Runtime helpers shared by the chess training and evaluation entrypoints.

This module resolves device, forward dtype, and DataLoader worker defaults in a
way that is safe for local CPU/MPS development and CUDA training.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


_VALID_DTYPE_NAMES = {"float16", "float32", "bfloat16"}


@dataclass(frozen=True)
class ResolvedRuntime:
    device: torch.device
    forward_dtype: torch.dtype
    forward_dtype_name: str
    num_workers: int
    warnings: Tuple[str, ...] = ()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_str)


def _normalize_dtype_name(dtype_name) -> str:
    if not isinstance(dtype_name, str):
        dtype_name = str(dtype_name).replace("torch.", "")
    return dtype_name.lower()


def _auto_forward_dtype_name(config, device: torch.device) -> str:
    preferred = _normalize_dtype_name(getattr(config, "forward_dtype", "float32"))
    if device.type == "cuda":
        if preferred in {"float16", "bfloat16"}:
            return preferred
        return "bfloat16"
    if device.type == "mps":
        return "float32"
    return "float32"


def _resolve_forward_dtype_name(
    config,
    device: torch.device,
    forward_dtype_str: Optional[str],
) -> Tuple[str, Tuple[str, ...]]:
    requested = _normalize_dtype_name(
        forward_dtype_str
        if forward_dtype_str is not None
        else getattr(config, "forward_dtype", "float32")
    )
    warnings = []

    if requested == "auto":
        return _auto_forward_dtype_name(config, device), tuple(warnings)

    if requested not in _VALID_DTYPE_NAMES:
        raise ValueError(
            f"Unsupported forward_dtype '{requested}'. "
            f"Expected one of: auto, {', '.join(sorted(_VALID_DTYPE_NAMES))}"
        )

    if device.type == "cpu" and requested != "float32":
        warnings.append(
            f"Requested forward_dtype={requested} on CPU; using float32 for stability."
        )
        return "float32", tuple(warnings)

    if device.type == "mps" and requested == "float16":
        warnings.append(
            "Requested forward_dtype=float16 on MPS. This is experimental on some "
            "Apple/PyTorch stacks and can crash in matmul kernels; prefer auto/float32 "
            "if you see MPSNDArrayMatrixMultiplication failures."
        )
        return "float16", tuple(warnings)

    if device.type == "mps" and requested == "bfloat16":
        warnings.append(
            "Requested forward_dtype=bfloat16 on MPS; using float32 instead."
        )
        return "float32", tuple(warnings)

    return requested, tuple(warnings)


def _resolve_num_workers(
    device: torch.device,
    num_workers: Optional[int],
) -> Tuple[int, Tuple[str, ...]]:
    warnings = []
    if num_workers is None:
        return (4 if device.type == "cuda" else 0), tuple(warnings)
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if num_workers > 0 and device.type in ("cpu", "mps"):
        warnings.append(
            f"num_workers={num_workers} on {device.type} uses spawned DataLoader "
            "workers, which can increase RAM usage and may fail under shared-memory "
            "restrictions."
        )
    return num_workers, tuple(warnings)


def resolve_training_runtime(
    config,
    device_str: str = "auto",
    num_workers: Optional[int] = None,
    forward_dtype_str: Optional[str] = None,
) -> ResolvedRuntime:
    """
    Resolve the runtime used by training/eval loops and update
    config.forward_dtype before model construction so chess/HRM modules agree
    on the active dtype.
    """

    device = _resolve_device(device_str)
    dtype_name, dtype_warnings = _resolve_forward_dtype_name(
        config,
        device,
        forward_dtype_str=forward_dtype_str,
    )
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported forward_dtype: {dtype_name}")

    resolved_workers, worker_warnings = _resolve_num_workers(device, num_workers)
    config.forward_dtype = dtype_name

    return ResolvedRuntime(
        device=device,
        forward_dtype=getattr(torch, dtype_name),
        forward_dtype_name=dtype_name,
        num_workers=resolved_workers,
        warnings=dtype_warnings + worker_warnings,
    )


def log_runtime(runtime: ResolvedRuntime, logger: Callable[[str], None]) -> None:
    logger(
        "Runtime:"
        f" device={runtime.device}"
        f" forward_dtype={runtime.forward_dtype_name}"
        f" num_workers={runtime.num_workers}"
    )
    for warning in runtime.warnings:
        logger(f"Runtime warning: {warning}")
