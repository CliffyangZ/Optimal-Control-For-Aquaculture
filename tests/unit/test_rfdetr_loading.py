"""Test RF-DETR model loading and device handling."""

import os
import sys
import torch
import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from core.infer.rfdetr_detector import RFDETRDetector


@pytest.mark.skipif(not os.path.exists("models/rf-detr-fish.pth"), reason="RF-DETR model not found")
def test_rfdetr_loads_on_cpu():
    """Test that RF-DETR loads successfully on CPU."""
    detector = RFDETRDetector.load(
        "models/rf-detr-fish.pth",
        device="cpu",
        conf_threshold=0.25,
    )
    assert detector is not None
    assert detector._model is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not os.path.exists("models/rf-detr-fish.pth"), reason="RF-DETR model not found")
def test_rfdetr_loads_on_cuda():
    """Test that RF-DETR loads successfully on CUDA."""
    detector = RFDETRDetector.load(
        "models/rf-detr-fish.pth",
        device="cuda",
        conf_threshold=0.25,
    )
    assert detector is not None
    assert detector._model is not None
    # Check device was set in the internal ModelContext
    if hasattr(detector._model, 'model'):
        assert detector._model.model.device.type == "cuda"


@pytest.mark.skipif(not os.path.exists("models/rf-detr-fish.pth"), reason="RF-DETR model not found")
def test_rfdetr_inference_succeeds():
    """Test that RF-DETR inference runs without error."""
    detector = RFDETRDetector.load(
        "models/rf-detr-fish.pth",
        device="cpu",
        conf_threshold=0.25,
    )
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Run inference
    result = detector.detect(dummy_frame, frame_id=0)

    assert result is not None
    assert result.frame_id == 0
    assert result.source_width == 640
    assert result.source_height == 480
    assert result.inference_ms > 0
