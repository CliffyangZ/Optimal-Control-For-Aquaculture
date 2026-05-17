"""Server-side rendering helpers for ripple visualization.

These functions render an `ActivityResult` from the spatiotemporal ripple
detector (`core.ripple_algo.RippleDetector`) onto a BGR video frame. They are
not used by the FastAPI live overlay path (overlays are drawn in the browser
from JSON regions over the WebRTC stream); they are kept for offline / debug
rendering and as a reference implementation of the heatmap colormap.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import setting
from core.ripple_algo import ActivityResult


def _colormap_id(name: str) -> int:
    """Resolve a colormap by name with a safe fallback."""
    name = (name or "").upper()
    return getattr(cv2, f"COLORMAP_{name}", cv2.COLORMAP_VIRIDIS)


def render_heatmap_overlay(
    frame_bgr: np.ndarray,
    result: ActivityResult,
    alpha: float | None = None,
    colormap_name: str | None = None,
) -> np.ndarray:
    """Render the per-pixel ripple heatmap onto a BGR frame.

    Uses `ActivityResult.heatmap` (already a per-pixel float32 in [0, 1])
    and blends it into `frame_bgr` with alpha compositing.

    Args:
        frame_bgr: Input BGR image with shape (H, W, 3).
        result: ActivityResult returned by `RippleDetector.process()`.
        alpha: Blend weight for the heatmap layer in [0, 1]. Defaults to
            `setting.HEATMAP_ALPHA`.
        colormap_name: Override colormap name; defaults to
            `setting.HEATMAP_COLORMAP`.

    Returns:
        BGR frame with the heatmap blended in.
    """
    h, w = frame_bgr.shape[:2]
    heat = result.heatmap
    if heat.shape != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, _colormap_id(colormap_name or setting.HEATMAP_COLORMAP))
    a = float(setting.HEATMAP_ALPHA if alpha is None else alpha)
    a = max(0.0, min(1.0, a))
    return cv2.addWeighted(frame_bgr, 1.0 - a, colored, a, 0)


def render_grid_lines(
    frame_bgr: np.ndarray,
    result: ActivityResult,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    draw_centers: bool = True,
) -> np.ndarray:
    """Draw the R×C grid used for per-cell scoring on top of a BGR frame.

    Args:
        frame_bgr: Input BGR image.
        result: ActivityResult returned by `RippleDetector.process()`; the grid
            shape is inferred from `result.mid_score.shape`.
        color: BGR color tuple for the grid lines / centre dots.
        thickness: Grid line thickness in pixels.
        draw_centers: Whether to render a small filled circle at each cell
            centre.

    Returns:
        A new BGR frame with grid lines drawn on top.
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    grid_rows, grid_cols = result.mid_score.shape
    if grid_rows <= 0 or grid_cols <= 0:
        return frame

    cell_h = h / grid_rows
    cell_w = w / grid_cols

    for r in range(1, grid_rows):
        y = int(round(r * cell_h))
        cv2.line(frame, (0, y), (w, y), color, thickness)
    for c in range(1, grid_cols):
        x = int(round(c * cell_w))
        cv2.line(frame, (x, 0), (x, h), color, thickness)

    if draw_centers:
        for r in range(grid_rows):
            for c in range(grid_cols):
                cx = int(round((c + 0.5) * cell_w))
                cy = int(round((r + 0.5) * cell_h))
                cv2.circle(frame, (cx, cy), 3, color, -1)

    return frame
