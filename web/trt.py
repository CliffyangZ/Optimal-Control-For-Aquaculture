"""TensorRT acceleration helpers for RF-DETR inference.

Optimization priority (highest to lowest):
  1. torch_tensorrt backend  — requires `pip install torch-tensorrt tensorrt`
  2. torch.compile (inductor) — ships with PyTorch 2.x, uses Triton on CUDA
  3. RF-DETR optimize_for_inference only (no compile) — CPU / MPS fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probe (import-time, no side effects)
# ---------------------------------------------------------------------------

def _probe() -> tuple[bool, bool, str]:
    """Return (cuda_available, trt_available, reason)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, False, "torch not installed"

    import torch

    if not torch.cuda.is_available():
        return False, False, "CUDA not available"

    try:
        import tensorrt  # noqa: F401
        import torch_tensorrt  # noqa: F401
        return True, True, "torch_tensorrt available"
    except ImportError as exc:
        return True, False, f"torch_tensorrt not installed ({exc})"


_CUDA_OK, _TRT_OK, _TRT_REASON = _probe()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OptimizationInfo(NamedTuple):
    backend: str        # "tensorrt" | "inductor" | "none"
    dtype: str          # "float16" | "float32"
    compiled: bool


def detect_support() -> dict:
    return {
        "cuda": _CUDA_OK,
        "tensorrt": _TRT_OK,
        "reason": _TRT_REASON,
    }


def load_optimized_detector(
    model_path: str | Path,
    device: str,
    conf_threshold: float,
    *,
    use_trt: bool = True,
) -> tuple:
    """Load RFDETRDetector with the best available acceleration.

    Returns (detector, OptimizationInfo).
    """
    from core.infer.rfdetr_detector import RFDETRDetector

    resolved_device = _resolve_device(device)
    is_cuda = resolved_device.startswith("cuda")

    # Choose dtype and compile settings
    if is_cuda:
        dtype = "float16"
        compile_model = True
    else:
        dtype = "float32"
        compile_model = False  # torch.compile has limited benefit on CPU/MPS

    # Attempt torch_tensorrt backend if requested and available
    backend = "none"
    if use_trt and _TRT_OK and is_cuda:
        backend = _try_trt_backend()

    detector = RFDETRDetector.load(
        model_path,
        device=resolved_device,
        conf_threshold=conf_threshold,
        optimize=True,
        compile=compile_model,
        dtype=dtype,
    )

    if compile_model:
        backend = backend if backend == "tensorrt" else "inductor"

    info = OptimizationInfo(
        backend=backend,
        dtype=dtype,
        compiled=compile_model,
    )
    return detector, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _try_trt_backend() -> str:
    """Return 'tensorrt' if torch_tensorrt compile backend works, else 'inductor'."""
    try:
        import torch
        import torch_tensorrt  # noqa: F401

        # Verify the backend is registered
        backends = torch._dynamo.list_backends()  # type: ignore[attr-defined]
        if "tensorrt" in backends:
            return "tensorrt"
        logger.warning("torch_tensorrt installed but 'tensorrt' backend not found in dynamo backends; using inductor")
        return "inductor"
    except Exception as exc:
        logger.warning("TRT backend probe failed (%s); falling back to inductor", exc)
        return "inductor"
