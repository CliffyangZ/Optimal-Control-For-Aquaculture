"""UI rendering components for spatiotemporal analysis visualization."""

import numpy as np
import cv2
from config import setting
from core.models import ActivityHeatmap


def render_heatmap_overlay(frame_bgr, heatmap: ActivityHeatmap) -> np.ndarray:
    """Render activity heatmap overlay on video frame.

    Creates a colormap visualization of per-cell activity scores overlaid
    on the frame using addWeighted blending.

    Args:
        frame_bgr: Input BGR image (H x W x 3)
        heatmap: ActivityHeatmap with per-cell scores

    Returns:
        BGR frame with heatmap overlay blended
    """
    h, w = frame_bgr.shape[:2]
    grid_rows, grid_cols = heatmap.cells.shape

    # Create heatmap as float array [0, 1]
    heatmap_array = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for r in range(grid_rows):
        for c in range(grid_cols):
            heatmap_array[r, c] = heatmap.cells[r, c].smoothed_score

    # Scale to frame dimensions
    heatmap_scaled = cv2.resize(
        heatmap_array,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert to uint8 [0, 255]
    heatmap_uint8 = (heatmap_scaled * 255).astype(np.uint8)

    # Apply colormap
    colormap_name = getattr(cv2, f"COLORMAP_{setting.HEATMAP_COLORMAP}")
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_name)

    # Blend with original frame
    alpha = setting.HEATMAP_ALPHA
    overlay = cv2.addWeighted(
        frame_bgr, 1.0 - alpha,
        heatmap_colored, alpha,
        0
    )

    return overlay


def render_grid_lines(frame_bgr, heatmap: ActivityHeatmap, color=(0, 255, 0), thickness=1) -> np.ndarray:
    """Draw grid cell boundary lines on frame.

    Args:
        frame_bgr: Input BGR image (H x W x 3)
        heatmap: ActivityHeatmap with grid dimensions
        color: Line color (BGR tuple)
        thickness: Line thickness in pixels

    Returns:
        BGR frame with grid lines drawn
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    grid_rows, grid_cols = heatmap.cells.shape

    cell_h = h // grid_rows
    cell_w = w // grid_cols

    # Draw horizontal lines
    for r in range(1, grid_rows):
        y = r * cell_h
        cv2.line(frame, (0, y), (w, y), color, thickness)

    # Draw vertical lines
    for c in range(1, grid_cols):
        x = c * cell_w
        cv2.line(frame, (x, 0), (x, h), color, thickness)

    # Draw cell center markers
    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = heatmap.cells[r, c]
            center_x = cell.center_x
            center_y = cell.center_y

            # Draw small circle at cell center
            radius = 3
            cv2.circle(frame, (center_x, center_y), radius, color, -1)

    return frame
