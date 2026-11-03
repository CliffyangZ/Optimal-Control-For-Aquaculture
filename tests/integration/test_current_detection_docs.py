"""Documentation checks for current YOLO detection guidance."""

from pathlib import Path


def test_root_readme_declares_current_yolo_source_of_truth() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "YOLO Web Detection" in content
    assert "detections.v1" in content
    assert "not the active web detection behavior" in content


def test_web_readme_documents_compatibility_overlay_names() -> None:
    content = Path("src/web/README.md").read_text(encoding="utf-8")

    assert "approved YOLO object detection" in content
    assert "overlay_ws_url" in content
    assert "last_overlay_at" in content
    assert "stale_overlay_ms" in content
