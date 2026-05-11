"""Spatiotemporal ripple detection — v3.

Design overview:

  0. Camera motion compensation (new):
                    KLT sparse optical flow → RANSAC homography H.
                    prev_gray and prev_edges are warped into the current
                    coordinate frame before any inter-frame comparison.
                    Whole-frame translation has div=0, so residual camera shake
                    does not pollute the divergence field.
  1. Preprocess  : gray + blur + CLAHE
  2. Specular mask: HSV-based suppression of water highlights
  3. MOG2         : adaptive learning rate (freezes when global activity is high)
  4. Soft fg mask : 0.3 base + 0.7 * (fg/255), so background is *attenuated*,
                    not silenced, and persistent water texture isn't fully eaten
                    by the background model.
  5. Optical flow : DIS PRESET_FAST on aligned frames.
                    Divergence (∂u/∂x + ∂v/∂y, positive part) is smoothed and
                    used as the primary candidate source.
  6. Candidate proposal (new):
                    div_smooth = GaussianBlur(div_pos, σ=div_peak_sigma)
                    local maxima above threshold → candidate ripple centres.
                    Replaces motion_mask | edge_xor → connected components.
  7. Temporal tracking (new):
                    Greedy nearest-neighbour matching across frames.
                    Only candidates persistent ≥ tracker_min_age frames are
                    passed to scoring — short-lived water-surface flashes drop out.
  8. Per-candidate structure:
                    - spatial  : radius-bin peak / top-k of radial edge alignment
                    - radial_flow + divergence : "outward + diverging" both required
                    - structure = w_spatial*spatial + w_expansion*expansion
  9. Heatmap render: each candidate is a Gaussian blob of amplitude=structure.
                    Pixel-wise max over candidates → ripple_heatmap.
 10. Grid pooling : structure_grid = max-pool ripple_heatmap over the R×C grid.
 11. Activity gate (per-cell):
                    motion/edge now computed on homography-aligned frames so
                    camera shake doesn't inflate the density metrics.
                    activity_grid = 0.5 * motion_soft + 0.5 * edge_soft
 12. Fusion      : raw_score = activity_grid * structure_grid
 13. Sliding windows: short / mid / long deques per cell (unchanged API).

Public surface unchanged: RippleAlgoConfig, ActivityResult, RippleDetector,
activity_level, run_on_video.  Legacy helpers `_radial_edge_score`,
`_radial_expansion_score` are kept as primitives.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RippleAlgoConfig:
    # -- Preprocess --------------------------------------------------------
    # NB: order matters.  v2 default is CLAHE *before* any blur so contrast
    # enhancement sees the raw high-frequency signal.  Canny already has its
    # own internal 5×5 Gaussian, so the explicit pre-blur is kept small
    # (or 0 to skip entirely) to avoid double-smoothing subtle ripples.
    blur_kernel: int = 3                    # set 0 or 1 to skip the explicit pre-blur
    clahe_clip: float = 3.0
    clahe_tile: int = 8
    clahe_before_blur: bool = True
    input_scale: float = 1.0

    # -- Visualization ----------------------------------------------------
    heatmap_colormap: int = cv2.COLORMAP_VIRIDIS  # cv2.COLORMAP_JET, VIRIDIS, INFERNO, etc.

    # -- Specular highlight suppression -----------------------------------
    use_specular_mask: bool = True
    specular_v_thresh: int = 220
    specular_s_thresh: int = 50
    specular_dilate: int = 5

    # -- MOG2 background subtraction --------------------------------------
    bg_history: int = 500
    bg_var_threshold: float = 16.0
    bg_dilate: int = 5
    bg_learning_rate: float = -1.0          # -1 → MOG2 default
    bg_freeze_activity: float = 0.30        # freeze updates when activity above this
    bg_soft_floor: float = 0.30             # base weight for background pixels

    # -- Edges -------------------------------------------------------------
    # Lower thresholds catch fainter ripple edges; relies on the radial-bin
    # peak score + MOG2 + structure-gating to suppress the extra noise.
    canny_low: int = 40
    canny_high: int = 120

    # -- Radial ring -------------------------------------------------------
    r_min: int = 8
    r_max: int = 120
    radius_bins: int = 8
    bin_min_edges: int = 6                  # bin needs at least this many edges
    bin_topk: int = 2                       # average of top-k bin scores

    # -- Optical flow ------------------------------------------------------
    flow_scale: float = 0.5
    dis_preset: int = 1                     # cv2.DISOpticalFlow_PRESET_FAST
    flow_mag_threshold: float = 0.12        # per-pixel noise floor
    divergence_cap: float = 0.20            # soft cap for div ring score

    # -- Dynamic evidence / candidate proposal ----------------------------
    diff_threshold: int = 6
    candidate_close_kernel: int = 5
    candidate_min_area: int = 25
    candidate_max_count: int = 16
    candidate_sigma: float = 18.0           # Gaussian blob sigma (px) for render
    candidate_min_score: float = 0.05

    # -- Camera motion compensation ----------------------------------------
    use_homography: bool = True
    klt_max_corners: int = 200
    klt_quality_level: float = 0.01
    klt_min_distance: int = 10
    ransac_reproj_thresh: float = 5.0
    homography_min_inliers: int = 20

    # -- Divergence peak detection -----------------------------------------
    div_peak_sigma: float = 8.0             # smoothing σ before peak search
    div_peak_threshold: float = 0.05        # min div value for a candidate

    # -- Temporal tracking -------------------------------------------------
    tracker_max_dist: int = 50              # max pixel gap to link across frames
    tracker_min_age: int = 3               # frames a peak must persist to pass

    # -- Soft saturation caps (replace v1 hard caps) ----------------------
    motion_cap: float = 0.03
    edge_cap: float = 0.03

    # -- Grid --------------------------------------------------------------
    grid_rows: int = 8
    grid_cols: int = 8

    # -- Sliding windows --------------------------------------------------
    short_window: int = 30
    mid_window: int = 150
    long_window: int = 900

    # -- Fusion ------------------------------------------------------------
    w_radial_flow: float = 0.6              # weight of outward flow inside "expansion"
    w_divergence: float = 0.4               # weight of flow divergence inside "expansion"
    w_motion: float = 0.5                   # weight of motion inside "activity"
    w_edge: float = 0.5                     # weight of edge change inside "activity"
    w_spatial: float = 0.6                  # weight of spatial inside "structure" (linear mode)
    w_expansion: float = 0.4                # weight of expansion inside "structure" (linear mode)
    structure_combine: str = "linear"       # "linear" | "geomean"

    # -- ROI / external mask ----------------------------------------------
    roi_mask: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ActivityResult:
    frame_index: int
    raw_score: np.ndarray            # (R, C) float32
    short_score: np.ndarray          # (R, C) float32
    mid_score: np.ndarray            # (R, C) float32
    long_score: np.ndarray           # (R, C) float32
    heatmap: np.ndarray              # (H, W) float32 in [0, 1]
    candidates: List[Tuple[int, int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def _preprocess(frame_bgr: np.ndarray, cfg: RippleAlgoConfig) -> np.ndarray:
    """Convert to gray, optionally blur, and apply CLAHE.

    Order is controlled by ``cfg.clahe_before_blur``:
      - True (default): gray → CLAHE → mild blur  (preserves fine ripple structure
                                                   that the contrast step would
                                                   otherwise never see)
      - False: gray → blur → CLAHE                (v1 behaviour, kept for A/B)

    ``cfg.blur_kernel ≤ 1`` skips the explicit blur entirely — useful because
    cv2.Canny already runs its own 5×5 Gaussian internally.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))

    def maybe_blur(img: np.ndarray) -> np.ndarray:
        if cfg.blur_kernel <= 1:
            return img
        k = cfg.blur_kernel | 1  # force odd
        return cv2.GaussianBlur(img, (k, k), 0)

    if cfg.clahe_before_blur:
        return maybe_blur(clahe.apply(gray))
    return clahe.apply(maybe_blur(gray))


def _specular_mask(frame_bgr: np.ndarray, cfg: RippleAlgoConfig) -> np.ndarray:
    """HSV-based mask of likely specular highlights (1 = highlight, 0 = clean)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = ((v >= cfg.specular_v_thresh) & (s <= cfg.specular_s_thresh)).astype(np.uint8)
    if cfg.specular_dilate > 1:
        k = np.ones((cfg.specular_dilate, cfg.specular_dilate), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def _canny_edges(gray: np.ndarray, cfg: RippleAlgoConfig) -> np.ndarray:
    return cv2.Canny(gray, cfg.canny_low, cfg.canny_high)


def _sobel_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _compute_flow_divergence(flow: np.ndarray) -> np.ndarray:
    """∂u/∂x + ∂v/∂y, clipped to its positive part."""
    u = flow[..., 0]
    v = flow[..., 1]
    du_dx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    dv_dy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    div = du_dx + dv_dy
    return np.maximum(div, 0.0)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def _ring_window(shape: Tuple[int, int], cx: int, cy: int,
                 r_max: int) -> Tuple[int, int, int, int]:
    H, W = shape
    y0 = max(0, cy - r_max)
    y1 = min(H, cy + r_max + 1)
    x0 = max(0, cx - r_max)
    x1 = min(W, cx + r_max + 1)
    return y0, y1, x0, x1


def _polar_coords(y0: int, y1: int, x0: int, x1: int,
                  cx: int, cy: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    r = np.sqrt(dx * dx + dy * dy)
    return dx, dy, r


def _soft_saturate(x: np.ndarray, cap: float) -> np.ndarray:
    """x / (x + cap) — smooth saturation, 0.5 at x=cap, no hard plateau."""
    return x / (x + cap + 1e-6)


# ---------------------------------------------------------------------------
# Radial scoring (per candidate centre)
# ---------------------------------------------------------------------------


def _radial_edge_score(edges: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                       cx: int, cy: int, cfg: RippleAlgoConfig) -> float:
    """Legacy: mean |r̂·g|/|g| over edge pixels in the full [r_min, r_max] ring."""
    r_min, r_max = cfg.r_min, cfg.r_max
    y0, y1, x0, x1 = _ring_window(edges.shape, cx, cy, r_max)
    if y1 <= y0 or x1 <= x0:
        return 0.0

    sub_e = edges[y0:y1, x0:x1]
    sub_gx = gx[y0:y1, x0:x1]
    sub_gy = gy[y0:y1, x0:x1]
    dx, dy, r = _polar_coords(y0, y1, x0, x1, cx, cy)

    in_ring = (r >= r_min) & (r <= r_max) & (sub_e > 0)
    if not np.any(in_ring):
        return 0.0

    inv_r = 1.0 / (r + 1e-6)
    rx = dx * inv_r
    ry = dy * inv_r
    g_mag = np.sqrt(sub_gx * sub_gx + sub_gy * sub_gy) + 1e-6
    alignment = np.abs(rx * sub_gx + ry * sub_gy) / g_mag
    return float(np.clip(alignment[in_ring].mean(), 0.0, 1.0))


def _radius_bin_peak_edge_score(edges: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                                cx: int, cy: int, cfg: RippleAlgoConfig) -> float:
    """V2 spatial: top-k mean across radius bins.

    Splits [r_min, r_max] into `radius_bins` equal-width bands; each band's
    mean radial alignment is computed independently.  The final score is the
    mean of the top-`bin_topk` bands so that one strong ring isn't diluted
    by the empty/noisy bands around it.
    """
    r_min, r_max = cfg.r_min, cfg.r_max
    n_bins = max(1, cfg.radius_bins)
    y0, y1, x0, x1 = _ring_window(edges.shape, cx, cy, r_max)
    if y1 <= y0 or x1 <= x0:
        return 0.0

    sub_e = edges[y0:y1, x0:x1]
    edge_mask = sub_e > 0
    if not edge_mask.any():
        return 0.0

    sub_gx = gx[y0:y1, x0:x1]
    sub_gy = gy[y0:y1, x0:x1]
    dx, dy, r = _polar_coords(y0, y1, x0, x1, cx, cy)

    inv_r = 1.0 / (r + 1e-6)
    rx = dx * inv_r
    ry = dy * inv_r
    g_mag = np.sqrt(sub_gx * sub_gx + sub_gy * sub_gy) + 1e-6
    alignment = np.abs(rx * sub_gx + ry * sub_gy) / g_mag

    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    scores = np.zeros(n_bins, dtype=np.float32)
    valid = np.zeros(n_bins, dtype=bool)
    for i in range(n_bins):
        b0, b1 = bin_edges[i], bin_edges[i + 1]
        m = (r >= b0) & (r < b1) & edge_mask
        n = int(m.sum())
        if n < cfg.bin_min_edges:
            continue
        scores[i] = alignment[m].mean()
        valid[i] = True

    if not valid.any():
        return 0.0
    valid_scores = scores[valid]
    if cfg.bin_topk <= 1 or valid_scores.size <= 1:
        return float(np.clip(valid_scores.max(), 0.0, 1.0))
    k = min(cfg.bin_topk, valid_scores.size)
    return float(np.clip(np.sort(valid_scores)[-k:].mean(), 0.0, 1.0))


def _radial_expansion_score(flow: np.ndarray, cx: int, cy: int,
                            cfg: RippleAlgoConfig) -> float:
    """Mean outward-projected (mag-weighted) flow over the ring."""
    r_min, r_max = cfg.r_min, cfg.r_max
    H, W = flow.shape[:2]
    y0, y1, x0, x1 = _ring_window((H, W), cx, cy, r_max)
    if y1 <= y0 or x1 <= x0:
        return 0.0

    sub = flow[y0:y1, x0:x1]
    u = sub[..., 0]
    v = sub[..., 1]
    dx, dy, r = _polar_coords(y0, y1, x0, x1, cx, cy)

    in_ring = (r >= r_min) & (r <= r_max)
    if not in_ring.any():
        return 0.0

    inv_r = 1.0 / (r + 1e-6)
    rx = dx * inv_r
    ry = dy * inv_r
    mag = np.sqrt(u * u + v * v)
    inv_m = 1.0 / (mag + 1e-6)
    u_unit = u * inv_m
    v_unit = v * inv_m

    outward = np.maximum(rx * u_unit + ry * v_unit, 0.0)
    weight = np.clip(mag / 2.0, 0.0, 1.0)
    weight = np.where(mag > cfg.flow_mag_threshold, weight, 0.0)
    score = (outward * weight)[in_ring].mean()
    return float(np.clip(score, 0.0, 1.0))


def _ring_pool_mean(field: np.ndarray, cx: int, cy: int,
                    r_min: int, r_max: int) -> float:
    H, W = field.shape
    y0, y1, x0, x1 = _ring_window((H, W), cx, cy, r_max)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    sub = field[y0:y1, x0:x1]
    _, _, r = _polar_coords(y0, y1, x0, x1, cx, cy)
    m = (r >= r_min) & (r <= r_max)
    if not m.any():
        return 0.0
    return float(sub[m].mean())


# ---------------------------------------------------------------------------
# Candidate proposal & heatmap rendering
# ---------------------------------------------------------------------------



def _estimate_homography(
    prev_gray: np.ndarray, gray: np.ndarray, cfg: RippleAlgoConfig
) -> Optional[np.ndarray]:
    """KLT + RANSAC homography mapping prev_gray → gray coordinate frame.

    Returns None when camera motion is negligible or the fit is unreliable.
    """
    corners = cv2.goodFeaturesToTrack(
        prev_gray, cfg.klt_max_corners, cfg.klt_quality_level, cfg.klt_min_distance
    )
    if corners is None or len(corners) < 8:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, corners, None)
    if status is None:
        return None
    good_prev = corners[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]
    if len(good_prev) < 8:
        return None
    homo, inlier_mask = cv2.findHomography(
        good_prev, good_next, cv2.RANSAC, cfg.ransac_reproj_thresh
    )
    if homo is None:
        return None
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if n_inliers < cfg.homography_min_inliers:
        return None
    return homo


def _find_divergence_peaks(
    div_smooth: np.ndarray, cfg: RippleAlgoConfig
) -> List[Tuple[int, int, float]]:
    """Local maxima of a pre-smoothed divergence field above threshold.

    Non-maximum suppression via 3×3 dilation then connectedComponents to merge
    adjacent sub-pixel peaks. Returns (cx, cy, peak_div) sorted high→low.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(div_smooth, kernel)
    local_max = (
        (div_smooth >= dilated) & (div_smooth > cfg.div_peak_threshold)
    ).astype(np.uint8)
    if not local_max.any():
        return []
    n, labels, _, centroids = cv2.connectedComponentsWithStats(
        local_max, connectivity=8
    )
    peaks: List[Tuple[int, int, float]] = []
    for i in range(1, n):
        cx_f, cy_f = centroids[i]
        peak_val = float(div_smooth[labels == i].max())
        peaks.append((int(round(cx_f)), int(round(cy_f)), peak_val))
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks[: cfg.candidate_max_count]


def _update_tracked_centers(
    new_centers: List[Tuple[int, int, float]],
    tracked: List[dict],
    max_dist: int,
    min_age: int,
) -> Tuple[List[dict], List[Tuple[int, int, float]]]:
    """Greedy nearest-neighbour tracking across frames.

    Returns (updated_state, stable_list) where stable_list contains only
    tracks that have been continuously matched for at least min_age frames.
    """
    matched_new = [False] * len(new_centers)
    updated: List[dict] = []

    for track in tracked:
        best_d = float(max_dist) + 1.0
        best_j = -1
        tcx, tcy = track["cx"], track["cy"]
        for j, (ncx, ncy, _) in enumerate(new_centers):
            if matched_new[j]:
                continue
            d = float(np.hypot(ncx - tcx, ncy - tcy))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            ncx, ncy, nval = new_centers[best_j]
            updated.append({"cx": ncx, "cy": ncy, "age": track["age"] + 1, "val": nval})
            matched_new[best_j] = True
        # unmatched tracks are dropped — ripple has dissipated

    for j, (ncx, ncy, nval) in enumerate(new_centers):
        if not matched_new[j]:
            updated.append({"cx": ncx, "cy": ncy, "age": 1, "val": nval})

    stable = [
        (t["cx"], t["cy"], t["val"]) for t in updated if t["age"] >= min_age
    ]
    return updated, stable


def _gaussian_patch(sigma: float) -> Tuple[np.ndarray, int]:
    """Pre-built Gaussian patch of amplitude 1 at centre. Returns (patch, half)."""
    half = max(1, int(np.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    g = np.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma)).astype(np.float32)
    return g, half


def _render_candidate_heatmap(scored: Sequence[Tuple[int, int, float]],
                              H: int, W: int, sigma: float) -> np.ndarray:
    """Pixel-wise max of amplitude=score Gaussian blobs at each (cx, cy)."""
    hm = np.zeros((H, W), dtype=np.float32)
    if not scored:
        return hm
    g, half = _gaussian_patch(sigma)
    for cx, cy, score in scored:
        if score <= 0:
            continue
        y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
        x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        gy0 = y0 - (cy - half); gy1 = gy0 + (y1 - y0)
        gx0 = x0 - (cx - half); gx1 = gx0 + (x1 - x0)
        patch = g[gy0:gy1, gx0:gx1] * float(score)
        view = hm[y0:y1, x0:x1]
        np.maximum(view, patch, out=view)
    return hm


def _grid_max_pool(field: np.ndarray, R: int, C: int) -> np.ndarray:
    H, W = field.shape
    out = np.zeros((R, C), dtype=np.float32)
    for r in range(R):
        y0 = int(round(r * H / R))
        y1 = int(round((r + 1) * H / R))
        if y1 <= y0:
            continue
        for c in range(C):
            x0 = int(round(c * W / C))
            x1 = int(round((c + 1) * W / C))
            if x1 <= x0:
                continue
            out[r, c] = float(field[y0:y1, x0:x1].max())
    return out


def _grid_density_pool(field: np.ndarray, R: int, C: int) -> np.ndarray:
    """Mean of a per-pixel density field over each grid cell."""
    H, W = field.shape
    out = np.zeros((R, C), dtype=np.float32)
    for r in range(R):
        y0 = int(round(r * H / R))
        y1 = int(round((r + 1) * H / R))
        if y1 <= y0:
            continue
        for c in range(C):
            x0 = int(round(c * W / C))
            x1 = int(round((c + 1) * W / C))
            if x1 <= x0:
                continue
            out[r, c] = float(field[y0:y1, x0:x1].mean())
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class RippleDetector:
    """Stateful spatiotemporal ripple detector (v2).

    Usage::

        det = RippleDetector(RippleAlgoConfig(grid_rows=8, grid_cols=8))
        for frame_bgr in stream:
            result = det.process(frame_bgr)
            if result is None:
                continue  # first frame
            heatmap = result.heatmap         # (H, W) float32 in [0, 1]
            score   = float(result.mid_score.mean())
            label   = activity_level(score)

    Call ``det.reset()`` between unrelated clips.
    """

    def __init__(self, cfg: Optional[RippleAlgoConfig] = None) -> None:
        self.cfg = cfg or RippleAlgoConfig()
        self._reset_state()

    # -- state -----------------------------------------------------------

    def _reset_state(self) -> None:
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_edges: Optional[np.ndarray] = None
        self._mog2 = None
        self._dis = None
        self._frame_index: int = -1
        self._windows = None
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._roi_resized: Optional[np.ndarray] = None
        self._last_activity: float = 0.0
        self._tracked_centers: List[dict] = []

    def reset(self) -> None:
        self._reset_state()

    # -- lazy initialisers ----------------------------------------------

    def _ensure_geometry(self, H: int, W: int) -> None:
        if self._cached_shape == (H, W):
            return
        R, C = self.cfg.grid_rows, self.cfg.grid_cols
        self._windows = [[
            {
                "short": deque(maxlen=self.cfg.short_window),
                "mid": deque(maxlen=self.cfg.mid_window),
                "long": deque(maxlen=self.cfg.long_window),
            }
            for _ in range(C)
        ] for _ in range(R)]
        if self.cfg.roi_mask is not None:
            self._roi_resized = cv2.resize(
                self.cfg.roi_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            self._roi_resized = None
        self._cached_shape = (H, W)

    def _ensure_mog2(self) -> None:
        if self._mog2 is None:
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=self.cfg.bg_history,
                varThreshold=self.cfg.bg_var_threshold,
                detectShadows=False,
            )

    def _ensure_dis(self) -> None:
        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(self.cfg.dis_preset)

    # -- optical flow ----------------------------------------------------

    def _optical_flow(self, prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
        self._ensure_dis()
        cfg = self.cfg
        if cfg.flow_scale != 1.0:
            h, w = gray.shape
            sw = max(16, int(w * cfg.flow_scale))
            sh = max(16, int(h * cfg.flow_scale))
            prev_s = cv2.resize(prev_gray, (sw, sh))
            cur_s = cv2.resize(gray, (sw, sh))
            flow_s = self._dis.calc(prev_s, cur_s, None)
            flow = cv2.resize(flow_s, (w, h))
            flow[..., 0] /= cfg.flow_scale
            flow[..., 1] /= cfg.flow_scale
            return flow
        return self._dis.calc(prev_gray, gray, None)

    # -- main entry ------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> Optional[ActivityResult]:
        cfg = self.cfg
        if cfg.input_scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None,
                                   fx=cfg.input_scale, fy=cfg.input_scale)

        gray = _preprocess(frame_bgr, cfg)
        H, W = gray.shape
        self._ensure_geometry(H, W)
        self._ensure_mog2()

        # ---- Specular + ROI valid mask --------------------------------
        if cfg.use_specular_mask:
            spec = _specular_mask(frame_bgr, cfg)
            valid_mask = (spec == 0).astype(np.float32)
        else:
            valid_mask = np.ones((H, W), dtype=np.float32)
        if self._roi_resized is not None:
            valid_mask *= (self._roi_resized > 0).astype(np.float32)

        # ---- MOG2 with adaptive learning rate -------------------------
        if self._last_activity >= cfg.bg_freeze_activity:
            lr = 0.0
        else:
            lr = cfg.bg_learning_rate
        fg_raw = self._mog2.apply(frame_bgr, learningRate=lr)
        if cfg.bg_dilate > 0:
            k = np.ones((cfg.bg_dilate, cfg.bg_dilate), np.uint8)
            fg_raw = cv2.dilate(fg_raw, k)
        # Soft fg: background pixels still count at bg_soft_floor weight.
        soft_fg = cfg.bg_soft_floor + (1.0 - cfg.bg_soft_floor) * (fg_raw.astype(np.float32) / 255.0)
        soft_fg *= valid_mask
        soft_fg = np.clip(soft_fg, 0.0, 1.0)

        # ---- Edges (used in evidence and per-candidate scoring) -------
        edges = _canny_edges(gray, cfg)

        # ---- First frame: store and return ----------------------------
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_edges = edges
            self._frame_index = 0
            return None

        self._frame_index += 1

        gx, gy = _sobel_gradients(gray)

        # ---- Camera motion compensation --------------------------------
        homo = None
        if cfg.use_homography:
            homo = _estimate_homography(self._prev_gray, gray, cfg)

        if homo is not None:
            prev_aligned = cv2.warpPerspective(self._prev_gray, homo, (W, H))
            prev_edges_aligned = cv2.warpPerspective(
                self._prev_edges, homo, (W, H), flags=cv2.INTER_NEAREST
            )
        else:
            prev_aligned = self._prev_gray
            prev_edges_aligned = self._prev_edges

        # ---- Inter-frame signals on aligned frames --------------------
        diff = cv2.absdiff(gray, prev_aligned)
        motion_mask = diff > cfg.diff_threshold
        edge_xor = np.bitwise_xor(edges, prev_edges_aligned) > 0

        motion_field = motion_mask.astype(np.float32) * soft_fg
        edge_field = edge_xor.astype(np.float32) * soft_fg

        # ---- Optical flow on aligned frames ---------------------------
        flow = self._optical_flow(prev_aligned, gray)
        div_pos = _compute_flow_divergence(flow)

        # ---- Divergence peak detection + temporal tracking ------------
        div_smooth = cv2.GaussianBlur(div_pos, (0, 0), cfg.div_peak_sigma)
        div_peaks = _find_divergence_peaks(div_smooth, cfg)
        self._tracked_centers, stable_peaks = _update_tracked_centers(
            div_peaks, self._tracked_centers, cfg.tracker_max_dist, cfg.tracker_min_age
        )
        candidates = stable_peaks  # [(cx, cy, div_val)]

        # ---- Score each candidate ------------------------------------
        scored: List[Tuple[int, int, float]] = []
        edges_for_scoring = (edges * (soft_fg > 0.0).astype(np.uint8)).astype(np.uint8)
        for cx, cy, _area in candidates:
            spatial = _radius_bin_peak_edge_score(edges_for_scoring, gx, gy, cx, cy, cfg)
            radial_flow = _radial_expansion_score(flow, cx, cy, cfg)
            div_ring = _ring_pool_mean(div_pos, cx, cy, cfg.r_min, cfg.r_max)
            div_norm = float(_soft_saturate(np.asarray(div_ring, dtype=np.float32),
                                            cfg.divergence_cap))
            wsum_exp = max(cfg.w_radial_flow + cfg.w_divergence, 1e-6)
            expansion = (cfg.w_radial_flow * radial_flow
                         + cfg.w_divergence * div_norm) / wsum_exp
            if cfg.structure_combine == "geomean":
                structure = float(np.sqrt(max(spatial, 0.0) * max(expansion, 0.0) + 1e-9))
            else:  # "linear" (default)
                wsum_s = max(cfg.w_spatial + cfg.w_expansion, 1e-6)
                structure = float(
                    (cfg.w_spatial * spatial + cfg.w_expansion * expansion) / wsum_s
                )
            if structure < cfg.candidate_min_score:
                continue
            scored.append((cx, cy, structure))

        # ---- Render candidate heatmap & pool to grid ------------------
        structure_hm = _render_candidate_heatmap(scored, H, W, cfg.candidate_sigma)
        R, C = cfg.grid_rows, cfg.grid_cols
        structure_grid = _grid_max_pool(structure_hm, R, C)

        # ---- Per-cell activity densities (soft saturated) ------------
        motion_density = _grid_density_pool(motion_field, R, C)
        edge_density = _grid_density_pool(edge_field, R, C)
        motion_soft = _soft_saturate(motion_density, cfg.motion_cap)
        edge_soft = _soft_saturate(edge_density, cfg.edge_cap)
        activity_grid = (cfg.w_motion * motion_soft + cfg.w_edge * edge_soft).astype(np.float32)
        # Normalise so weights summing past 1 don't push above 1
        wsum = max(cfg.w_motion + cfg.w_edge, 1e-6)
        activity_grid /= wsum

        # ---- Gated fusion --------------------------------------------
        raw_score = (activity_grid * structure_grid).astype(np.float32)
        raw_score = np.clip(raw_score, 0.0, 1.0)

        # ---- Sliding windows -----------------------------------------
        short_s = np.zeros((R, C), dtype=np.float32)
        mid_s = np.zeros((R, C), dtype=np.float32)
        long_s = np.zeros((R, C), dtype=np.float32)
        for r in range(R):
            for c in range(C):
                w = self._windows[r][c]
                v = float(raw_score[r, c])
                w["short"].append(v)
                w["mid"].append(v)
                w["long"].append(v)
                short_s[r, c] = float(np.mean(w["short"]))
                mid_s[r, c] = float(np.mean(w["mid"]))
                long_s[r, c] = float(np.mean(w["long"]))

        # Final heatmap: bilinear-upsampled mid_score, modulated by structure
        # (keeps grid-consistency but sharpens around candidate centres).
        mid_full = cv2.resize(mid_s, (W, H), interpolation=cv2.INTER_LINEAR)
        # Smooth structure heatmap to blend nicely with bilinear mid_score
        struct_smooth = cv2.GaussianBlur(structure_hm, (0, 0), cfg.candidate_sigma * 0.6)
        heatmap = np.clip(0.6 * mid_full + 0.4 * struct_smooth, 0.0, 1.0).astype(np.float32)

        # ---- Update inter-frame state --------------------------------
        self._prev_gray = gray
        self._prev_edges = edges
        self._last_activity = float(activity_grid.mean())

        return ActivityResult(
            frame_index=self._frame_index,
            raw_score=raw_score,
            short_score=short_s,
            mid_score=mid_s,
            long_score=long_s,
            heatmap=heatmap,
            candidates=scored,
        )


# ---------------------------------------------------------------------------
# Activity labels
# ---------------------------------------------------------------------------


def activity_level(score: float) -> str:
    if score < 0.15:
        return "idle"
    if score < 0.30:
        return "low"
    if score < 0.50:
        return "medium"
    if score < 0.70:
        return "high"
    return "intense"


# Colormap suggestions for different purposes
# VIRIDIS (default): perceptually uniform, good for all eye types
# INFERNO: high contrast, emphasizes peaks well
# JET: traditional but has perceptual issues
# TURBO: modern alternative to JET, better perceptual uniformity
def get_colormap(name: str) -> int:
    """Get OpenCV colormap ID by name."""
    colormaps = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "spring": cv2.COLORMAP_SPRING,
        "summer": cv2.COLORMAP_SUMMER,
        "autumn": cv2.COLORMAP_AUTUMN,
        "winter": cv2.COLORMAP_WINTER,
        "bone": cv2.COLORMAP_BONE,
        "rainbow": cv2.COLORMAP_RAINBOW,
    }
    return colormaps.get(name.lower(), cv2.COLORMAP_VIRIDIS)


_LEVEL_COLOR_BGR = {
    "idle": (128, 128, 128),
    "low": (200, 200, 0),
    "medium": (0, 220, 220),
    "high": (0, 140, 255),
    "intense": (0, 0, 255),
}


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def run_on_video(
    video_path: str,
    cfg: Optional[RippleAlgoConfig] = None,
    show: bool = True,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    log_every: int = 30,
    draw_candidates: bool = True,
) -> None:
    """Run RippleDetector on a video file with optional preview / save."""

    cfg = cfg or RippleAlgoConfig()
    detector = RippleDetector(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    n = 0
    t_start = time.time()
    label = "idle"
    global_score = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = detector.process(frame)
            if result is None:
                n += 1
                continue

            disp_h, disp_w = result.heatmap.shape
            if (disp_h, disp_w) != frame.shape[:2]:
                frame = cv2.resize(frame, (disp_w, disp_h))

            overlay = cv2.applyColorMap(
                (result.heatmap * 255).astype(np.uint8),
                cfg.heatmap_colormap,
            )
            vis = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)

            # Per-cell dots coloured by mid_score level
            R, C = cfg.grid_rows, cfg.grid_cols
            for r in range(R):
                for c in range(C):
                    y0 = int(round(r * disp_h / R))
                    y1 = int(round((r + 1) * disp_h / R))
                    x0 = int(round(c * disp_w / C))
                    x1 = int(round((c + 1) * disp_w / C))
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    lvl = activity_level(float(result.mid_score[r, c]))
                    cv2.circle(vis, (cx, cy), 3, _LEVEL_COLOR_BGR[lvl], -1)

            # Candidate centres (white outline + score)
            if draw_candidates and result.candidates:
                for cx, cy, score in result.candidates:
                    cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(vis, f"{score:.2f}", (cx + 8, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

            peak_score = float(result.mid_score.max())
            mean_score = float(result.mid_score.mean())
            label = activity_level(peak_score)
            text = (f"{label}  peak={peak_score:.2f}  mean={mean_score:.2f}  "
                    f"f={result.frame_index}")
            cv2.putText(vis, text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        _LEVEL_COLOR_BGR[label], 2, cv2.LINE_AA)

            if output_path is not None:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (disp_w, disp_h))
                writer.write(vis)

            if show:
                cv2.imshow("ripple", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            n += 1
            if log_every > 0 and n % log_every == 0:
                elapsed = time.time() - t_start
                proc_fps = n / elapsed if elapsed > 0 else 0.0
                tot = f"/{total_frames}" if total_frames else ""
                print(
                    f"  frame {n}{tot}  peak={peak_score:.2f}  "
                    f"mean={mean_score:.2f}  level={label:<7}  "
                    f"cand={len(result.candidates):2d}  proc_fps={proc_fps:.1f}",
                    flush=True,
                )
            if max_frames is not None and n >= max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        elapsed = time.time() - t_start
        if elapsed > 0:
            print(f"Done. processed {n} frames in {elapsed:.1f}s "
                  f"({n / elapsed:.1f} fps)", flush=True)
        else:
            print(f"Done. processed {n} frames", flush=True)
