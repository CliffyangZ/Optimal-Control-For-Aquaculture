"""Spatiotemporal ripple detection — v4.

Design overview:

  0. Camera motion compensation:
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
  5. Optical flow (v4 — edge-driven):
                    DIS PRESET_FAST runs on a soft edge field
                    (dilated Canny → light Gaussian blur), not on raw gray.
                    Rationale: water-surface gray has a lot of low-texture and
                    specular gradient noise that DIS happily snaps onto,
                    polluting divergence with random outward components.  Soft
                    edges are sparse-but-continuous, so flow locks onto the
                    actual ripple wavefronts and the divergence field becomes
                    far cleaner.  Set ``cfg.flow_on_edges=False`` to revert.
                    Divergence (∂u/∂x + ∂v/∂y, positive part) is smoothed and
                    used as the primary candidate source.
  6. Candidate proposal:
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
 13. Sliding windows: short / mid / long ring buffers + running sums
                    (vectorised in v4; same public API: short_score, mid_score,
                    long_score arrays of shape (R, C)).

Public surface unchanged: RippleAlgoConfig, ActivityResult, RippleDetector,
activity_level, run_on_video.  Legacy helpers `_radial_edge_score`,
`_radial_expansion_score` are kept as primitives.
"""

from __future__ import annotations

import time
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
    # Soft saturation cap for divergence ring score: ring-pooled divergence
    # of a strong ripple is around 1–3, so we saturate at ~1.5 to keep the
    # candidate score informative without flat-topping.
    divergence_cap: float = 1.5
    # Edge-driven flow (v4):
    #   When True, optical flow is computed on a soft edge field (dilated Canny
    #   blurred with a small Gaussian) instead of raw gray.  This makes the
    #   divergence map much cleaner over featureless / specular water because
    #   DIS no longer hallucinates flow in low-texture regions.
    #
    #   Trade-off: edge-driven flow has smaller absolute divergence magnitudes
    #   (the input gradient itself is smaller), so ``div_peak_threshold`` and
    #   ``divergence_cap`` are scaled down by ``edge_flow_div_scale`` when
    #   this mode is active.  Tune that one knob to control sensitivity.
    flow_on_edges: bool = True
    edge_flow_dilate: int = 3               # px — thickens 1-pixel Canny edges
    edge_flow_blur_sigma: float = 1.2       # σ for soft-edge blur (0 → skip)
    # Per-frame percentile-driven thresholds (see div_peak_percentile) handle
    # the magnitude difference between gray-flow and edge-flow divergence
    # automatically, so the legacy fixed scale is no longer needed.  Kept
    # at 1.0 for back-compat; tune below 1 to bias edge mode toward more
    # candidates, above 1 for fewer.
    edge_flow_div_scale: float = 1.0

    # -- Dynamic evidence / candidate proposal ----------------------------
    diff_threshold: int = 6
    candidate_close_kernel: int = 5
    candidate_min_area: int = 25
    candidate_max_count: int = 16
    candidate_sigma: float = 18.0           # Gaussian blob sigma (px) for render
    candidate_min_score: float = 0.04       # was 0.05; let weak first-frame onsets through

    # -- Camera motion compensation ----------------------------------------
    use_homography: bool = True
    klt_max_corners: int = 200
    klt_quality_level: float = 0.01
    klt_min_distance: int = 10
    ransac_reproj_thresh: float = 5.0
    homography_min_inliers: int = 20
    # Camera motion is a low-frequency signal — running KLT+RANSAC every other
    # frame and reusing the cached H halves the homography cost with no
    # noticeable accuracy loss for handheld / mounted cameras.  Set to 1 to
    # estimate every frame, 0 to disable caching entirely.
    homography_stride: int = 2

    # -- Divergence peak detection -----------------------------------------
    div_peak_sigma: float = 8.0             # smoothing σ before peak search
    # Absolute floor: peaks must be at least this strong.  Divergence after
    # Sobel + σ=8 Gaussian sits around 0.1–0.5 on quiet water; real ripple
    # impacts spike to 3–10.  Floor is the minimum required even on a busy
    # scene where the adaptive term would otherwise relax it.
    div_peak_threshold: float = 0.8
    # Adaptive threshold:
    #   threshold = max(abs_floor, mean + k_std × std).
    # On a quiet pond mean+kσ < abs_floor → floor wins.  On a busy scene
    # (lots of ambient noise) the term rises above the floor and stops the
    # candidate pool from flooding.  Set div_peak_k_std=0 to disable.
    div_peak_k_std: float = 3.5
    # NMS radius for peak suppression — two peaks closer than this collapse
    # to the stronger one.  Default ≈ div_peak_sigma so we don't pick up
    # multiple sub-peaks inside a single ripple blob.
    div_peak_nms_radius: int = 12
    # Run peak search on a downsampled copy of div_smooth — div_smooth is
    # already heavily blurred so peaks survive a 2-3× downscale.  Empirical
    # speedup: 3-5× on the peak-finder stage for HD frames.  Set to 1.0 to
    # search at full resolution.
    div_peak_search_scale: float = 0.5

    # -- Temporal tracking -------------------------------------------------
    tracker_max_dist: int = 50              # max pixel gap to link across frames
    tracker_min_age: int = 3                # frames a peak must persist to pass

    # -- Impact-event fast path --------------------------------------------
    # A water droplet impact is a single-frame energy spike — by the time the
    # 3-frame tracker confirms it the splash is half-gone.  Peaks whose
    # *initial* divergence is strong enough (above impact_div_multiplier ×
    # current threshold) are promoted to "stable" on their first appearance,
    # so the heatmap lights up the moment the droplet hits the surface.
    # Tune impact_div_multiplier upward to be more selective (fewer false
    # promotions) and downward to be more sensitive.
    impact_event_enable: bool = True
    impact_div_multiplier: float = 6.0      # × div_peak_threshold → instant promote
    # Short pulse: max-over-last-N raw_score blended into the heatmap so a
    # 1–2-frame spike isn't washed out by the 30-frame short_window mean.
    # A higher pulse weight makes the heatmap react faster to droplet
    # impacts and other transient events at the cost of slight flicker on
    # continuous activity.
    short_pulse_frames: int = 5
    short_pulse_weight: float = 0.55        # weight in final heatmap blend

    # -- Frame-level energy aggregation ------------------------------------
    # ``energy`` is the scalar 0–1 measure of "how much ripple activity is
    # happening in this frame right now".  It's derived from the per-
    # candidate structure scores via a probabilistic-OR aggregation:
    #
    #     energy = 1 - ∏ (1 - score_i * energy_candidate_weight)
    #
    # Each candidate is treated as an independent "ripple event" whose
    # individual contribution is its structure score (already 0-1).  The
    # OR aggregation saturates smoothly toward 1 as more candidates pile
    # up — matching the intuition that 1 strong ripple < 3 strong ripples
    # without ever exceeding 1.0.
    #
    # ``energy_candidate_weight`` (k) scales each candidate before the
    # OR so a single candidate doesn't immediately push energy near 1:
    #   k=1.0 → 1 cand at score 0.6 yields energy 0.60
    #   k=0.7 → ...yields 0.42 (1 cand = "low–medium")
    #   k=0.5 → ...yields 0.30 (1 cand = "low")
    energy_candidate_weight: float = 0.9
    # Each candidate's contribution is also scaled by a confidence factor
    # derived from its raw divergence amplitude:
    #     div_conf = div_val / (div_val + energy_div_softcap)
    # softcap=3 means div_val=3 yields 0.5 confidence, div_val=10 yields
    # 0.77, div_val=30 yields 0.91.  This makes a strong splash (div ~16)
    # clearly outscore a passive ripple (div ~1.5) at the *same* structure
    # quality — the missing axis in the v4.2 energy.
    energy_div_softcap: float = 3.0

    # Asymmetric EMA (attack–release envelope) on the per-frame energy.
    # The water-droplet event "should look like" a bell curve — fast rise,
    # slow decay — but raw candidate detection is jittery (1-frame misses
    # are common during the splash and fading aftermath).  An envelope
    # follower with attack >> release smooths over those single-frame dips
    # without lagging behind real onsets:
    #
    #     if energy_raw > energy_ema:   # rising
    #         energy_ema = α_attack * energy_raw + (1-α_attack) * energy_ema
    #     else:                         # falling
    #         energy_ema = α_release * energy_raw + (1-α_release) * energy_ema
    #
    # Defaults at 30 fps:
    #   attack=0.6  → 75% in ~2 frames (event onsets feel instant)
    #   release=0.15 → 75% in ~8 frames (~0.3 s decay tail)
    # Set both to the same value for plain symmetric EMA.  Set both to 0
    # to fall back to the legacy ``energy_ema_alpha`` (single rate).
    energy_attack_alpha: float = 0.6
    energy_release_alpha: float = 0.15
    # Legacy single-rate EMA — used only when both attack and release are 0.
    energy_ema_alpha: float = 0.5

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
    # Scalar 0-1 "ripple energy" of the whole frame — probabilistic OR of
    # per-candidate structure scores, EMA-smoothed in time.  This is the
    # right number to log / display as "how active is the water surface
    # right now"; it's far more responsive than mid_score.mean() and far
    # less floor-pinned than mid_score.max().
    energy: float = 0.0
    # Pre-smoothing raw energy (this frame only, no EMA) for debugging.
    energy_raw: float = 0.0


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


def _edge_field_for_flow(edges: np.ndarray, cfg: RippleAlgoConfig) -> np.ndarray:
    """Turn a binary Canny edge map into a soft, DIS-friendly intensity field.

    DIS optical flow needs a smooth gradient signal to estimate displacement;
    a raw 1-pixel-wide binary edge has near-zero support outside the edge
    itself and gives DIS nothing to track.  We thicken edges with a small
    dilation then apply a light Gaussian to give each edge a few-pixel ramp,
    which is enough for DIS to lock its flow vectors onto wavefront motion.

    Output is uint8 in [0, 255] for compatibility with ``DISOpticalFlow.calc``.
    """
    if cfg.edge_flow_dilate > 1:
        k = np.ones((cfg.edge_flow_dilate, cfg.edge_flow_dilate), np.uint8)
        soft = cv2.dilate(edges, k, iterations=1)
    else:
        soft = edges.copy()
    if cfg.edge_flow_blur_sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), cfg.edge_flow_blur_sigma)
    # Stretch to full uint8 dynamic range so DIS sees high contrast.
    if soft.dtype != np.uint8:
        soft = np.clip(soft, 0, 255).astype(np.uint8)
    return soft


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


def _aggregate_candidate_energy(
    strengths: Sequence[float], weight: float
) -> float:
    """Probabilistic-OR aggregation of per-candidate strength scores.

        energy = 1 - ∏ (1 - clip(strength_i × weight, 0, 0.99))

    Each candidate is treated as an independent "is ripple here" event;
    OR-ing their probabilities gives a number that grows monotonically
    with both how *strong* each ripple is and how *many* ripples are
    happening, saturating gracefully toward 1.0 without ever exceeding it.

    ``strength_i`` is typically (structure_score × divergence_confidence)
    so the aggregation is sensitive to both ripple quality and amplitude.
    ``weight`` (typically 0.5–1.0) is a global gain knob on each event.
    """
    if not strengths:
        return 0.0
    not_p = 1.0
    for s in strengths:
        e = float(s) * weight
        if e <= 0.0:
            continue
        if e >= 0.99:
            e = 0.99
        not_p *= (1.0 - e)
    return float(1.0 - not_p)


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


_NMS_KERNEL_3 = np.ones((3, 3), np.uint8)
_NMS_KERNEL_CACHE: dict = {}


def _nms_kernel(radius: int) -> np.ndarray:
    """Disk-shaped structuring element, cached by radius."""
    radius = max(1, int(radius))
    k = _NMS_KERNEL_CACHE.get(radius)
    if k is None:
        size = 2 * radius + 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        _NMS_KERNEL_CACHE[radius] = k
    return k


def _find_divergence_peaks_at(
    div_smooth: np.ndarray, threshold: float, max_count: int,
    nms_radius: int = 1, search_scale: float = 1.0,
) -> List[Tuple[int, int, float]]:
    """Local maxima of a pre-smoothed divergence field above ``threshold``.

    Vectorised path:

      1. (Optional) downsample by ``search_scale`` for the NMS dilate, then
         scale coordinates back.  div_smooth was blurred at σ=8, so a 2-3×
         downsample loses no peak information but makes both the dilate and
         the boolean scan many times cheaper.
      2. Dilate with a disk of radius ``nms_radius`` → each pixel becomes
         the max over its local neighbourhood.  A pixel that equals its
         dilated value is the unique max within ``nms_radius`` px.
      3. mask = (div_smooth >= dilated) & (div_smooth > threshold).
      4. numpy ``argpartition`` picks the top-``max_count`` in O(N).

    Returns (cx, cy, peak_div) in the *original* image coordinates,
    sorted high→low.
    """
    if search_scale != 1.0 and 0.1 < search_scale < 1.0:
        H, W = div_smooth.shape
        sw, sh = max(8, int(W * search_scale)), max(8, int(H * search_scale))
        # cv2.resize with INTER_AREA preserves local maxima fairly well
        small = cv2.resize(div_smooth, (sw, sh), interpolation=cv2.INTER_AREA)
        # Scale NMS radius proportionally so the suppression neighbourhood
        # covers the same physical area.
        rad = max(1, int(round(nms_radius * search_scale)))
        kernel = _NMS_KERNEL_3 if rad <= 1 else _nms_kernel(rad)
        dilated = cv2.dilate(small, kernel)
        local_max = (small >= dilated) & (small > threshold)
        if not local_max.any():
            return []
        ys, xs = np.where(local_max)
        vals = small[ys, xs]
        if vals.size > max_count:
            top = np.argpartition(vals, -max_count)[-max_count:]
            ys, xs, vals = ys[top], xs[top], vals[top]
        order = np.argsort(-vals)
        # Map coordinates back to the full-resolution frame.
        inv = 1.0 / search_scale
        return [
            (int(round(xs[i] * inv)), int(round(ys[i] * inv)), float(vals[i]))
            for i in order
        ]

    kernel = _NMS_KERNEL_3 if nms_radius <= 1 else _nms_kernel(nms_radius)
    dilated = cv2.dilate(div_smooth, kernel)
    local_max = (div_smooth >= dilated) & (div_smooth > threshold)
    if not local_max.any():
        return []
    ys, xs = np.where(local_max)
    vals = div_smooth[ys, xs]
    if vals.size > max_count:
        top = np.argpartition(vals, -max_count)[-max_count:]
        ys, xs, vals = ys[top], xs[top], vals[top]
    order = np.argsort(-vals)
    return [
        (int(xs[i]), int(ys[i]), float(vals[i])) for i in order
    ]


def _find_divergence_peaks(
    div_smooth: np.ndarray, cfg: RippleAlgoConfig
) -> List[Tuple[int, int, float]]:
    """Back-compat shim that pulls threshold + max_count from cfg."""
    return _find_divergence_peaks_at(
        div_smooth, cfg.div_peak_threshold, cfg.candidate_max_count
    )


def _update_tracked_centers(
    new_centers: List[Tuple[int, int, float]],
    tracked: List[dict],
    max_dist: int,
    min_age: int,
    impact_val: float = float("inf"),
) -> Tuple[List[dict], List[Tuple[int, int, float]]]:
    """Greedy nearest-neighbour tracking across frames.

    Args:
      impact_val: any new candidate with ``val >= impact_val`` is promoted
                  to "stable" on its first appearance (age 1).  Use this to
                  surface single-frame impact events — e.g. a water droplet
                  splash — without waiting for ``min_age`` confirmations.

    Returns (updated_state, stable_list) where stable_list contains tracks
    that either reached ``min_age`` *or* were impact-promoted this frame.
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

    stable: List[Tuple[int, int, float]] = []
    for t in updated:
        if t["age"] >= min_age or t["val"] >= impact_val:
            stable.append((t["cx"], t["cy"], t["val"]))
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


def _crop_to_grid(field: np.ndarray, R: int, C: int) -> np.ndarray:
    """Crop to the largest multiple of (R, C) so we can reshape-pool cleanly.

    The lost border (≤ R-1 rows / C-1 cols) is negligible for 8×8 grids on
    HD frames and avoids the per-cell Python loop in pooling.
    """
    H, W = field.shape
    Hc = (H // R) * R
    Wc = (W // C) * C
    return field[:Hc, :Wc]


def _grid_max_pool(field: np.ndarray, R: int, C: int) -> np.ndarray:
    """Vectorised max-pool of a 2-D field into an (R, C) grid."""
    f = _crop_to_grid(field, R, C)
    if f.size == 0:
        return np.zeros((R, C), dtype=np.float32)
    H, W = f.shape
    bh, bw = H // R, W // C
    # Reshape into (R, bh, C, bw) and max over the bh / bw axes.
    return f.reshape(R, bh, C, bw).max(axis=(1, 3)).astype(np.float32)


def _grid_density_pool(field: np.ndarray, R: int, C: int) -> np.ndarray:
    """Vectorised mean-pool (per-pixel density) into an (R, C) grid."""
    f = _crop_to_grid(field, R, C)
    if f.size == 0:
        return np.zeros((R, C), dtype=np.float32)
    H, W = f.shape
    bh, bw = H // R, W // C
    return f.reshape(R, bh, C, bw).mean(axis=(1, 3)).astype(np.float32)


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
        self._windows = None                 # legacy, kept for back-compat
        self._sw: Optional[dict] = None      # v4 vectorised sliding windows
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._roi_resized: Optional[np.ndarray] = None
        self._last_activity: float = 0.0
        self._tracked_centers: List[dict] = []
        # v4.1: cached homography (so we can stride KLT) + short pulse buffer
        self._homo_cache: Optional[np.ndarray] = None
        self._pulse_buf: Optional[np.ndarray] = None  # (N, R, C) ring buffer
        self._pulse_idx: int = 0
        self._pulse_count: int = 0
        # v4.3: EMA-smoothed frame-level energy
        self._energy_ema: float = 0.0

    def reset(self) -> None:
        self._reset_state()

    # -- lazy initialisers ----------------------------------------------

    def _ensure_geometry(self, H: int, W: int) -> None:
        if self._cached_shape == (H, W):
            return
        R, C = self.cfg.grid_rows, self.cfg.grid_cols
        # Vectorised sliding windows: ring buffers + running sums per scale.
        # Shape (window_len, R, C); we overwrite the oldest slot each frame
        # and keep a running float64 sum so per-frame mean is O(R*C).
        self._sw = {
            "short": {
                "buf": np.zeros((self.cfg.short_window, R, C), dtype=np.float32),
                "sum": np.zeros((R, C), dtype=np.float64),
                "len": self.cfg.short_window,
                "count": 0,
                "idx": 0,
            },
            "mid": {
                "buf": np.zeros((self.cfg.mid_window, R, C), dtype=np.float32),
                "sum": np.zeros((R, C), dtype=np.float64),
                "len": self.cfg.mid_window,
                "count": 0,
                "idx": 0,
            },
            "long": {
                "buf": np.zeros((self.cfg.long_window, R, C), dtype=np.float32),
                "sum": np.zeros((R, C), dtype=np.float64),
                "len": self.cfg.long_window,
                "count": 0,
                "idx": 0,
            },
        }
        # _windows kept as None — legacy attribute, unused after v4.
        self._windows = None
        # Short-pulse ring buffer for impact-event blending
        n = max(1, self.cfg.short_pulse_frames)
        self._pulse_buf = np.zeros((n, R, C), dtype=np.float32)
        self._pulse_idx = 0
        self._pulse_count = 0
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

    def _optical_flow(self, prev_img: np.ndarray, cur_img: np.ndarray) -> np.ndarray:
        """DIS optical flow with optional downscale.

        ``prev_img`` and ``cur_img`` are uint8 single-channel arrays of the
        same shape — either raw gray or a soft-edge field built by
        ``_edge_field_for_flow``.  Whichever pair is passed, the same pair
        is used; the caller picks the representation.
        """
        self._ensure_dis()
        cfg = self.cfg
        if cfg.flow_scale != 1.0:
            h, w = cur_img.shape
            sw = max(16, int(w * cfg.flow_scale))
            sh = max(16, int(h * cfg.flow_scale))
            prev_s = cv2.resize(prev_img, (sw, sh))
            cur_s = cv2.resize(cur_img, (sw, sh))
            flow_s = self._dis.calc(prev_s, cur_s, None)
            flow = cv2.resize(flow_s, (w, h))
            flow[..., 0] /= cfg.flow_scale
            flow[..., 1] /= cfg.flow_scale
            return flow
        return self._dis.calc(prev_img, cur_img, None)

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
        # Strided estimation: KLT+RANSAC every ``homography_stride`` frames,
        # reuse the cached H in between.  Camera motion is low-frequency so
        # a 2-frame cache is essentially free in accuracy and ~halves cost.
        homo = None
        if cfg.use_homography:
            stride = max(1, cfg.homography_stride)
            if (self._frame_index % stride) == 0:
                homo = _estimate_homography(self._prev_gray, gray, cfg)
                if homo is not None:
                    self._homo_cache = homo
            else:
                homo = self._homo_cache

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
        # v4: prefer edge-driven flow.  Soft-edge fields (dilated Canny +
        # light Gaussian) give DIS a sparse, high-contrast signal so flow
        # locks onto ripple wavefronts instead of low-texture water noise,
        # which makes the divergence map much cleaner.  Fall back to raw
        # gray when the caller opts out.
        if cfg.flow_on_edges:
            cur_for_flow = _edge_field_for_flow(edges, cfg)
            prev_for_flow = _edge_field_for_flow(prev_edges_aligned, cfg)
        else:
            cur_for_flow = gray
            prev_for_flow = prev_aligned
        flow = self._optical_flow(prev_for_flow, cur_for_flow)
        div_pos = _compute_flow_divergence(flow)

        # Edge-driven flow has lower-amplitude divergence than gray flow, but
        # the percentile-based threshold below adapts automatically.  The
        # explicit scale knob is kept for fine manual override.
        div_scale = (
            max(cfg.edge_flow_div_scale, 1e-3) if cfg.flow_on_edges else 1.0
        )
        divergence_cap = cfg.divergence_cap * div_scale

        # ---- Divergence peak detection + temporal tracking ------------
        div_smooth = cv2.GaussianBlur(div_pos, (0, 0), cfg.div_peak_sigma)
        # Adaptive threshold = max(absolute floor, mean + k·std).
        # Mean+k·std auto-tunes to scene activity (cheap statistical estimate
        # using cv2.meanStdDev — order of magnitude faster than np.percentile
        # on a full HD divergence map).
        abs_floor = cfg.div_peak_threshold * div_scale
        if cfg.div_peak_k_std > 0:
            m, s = cv2.meanStdDev(div_smooth)
            adaptive_floor = float(m[0, 0]) + cfg.div_peak_k_std * float(s[0, 0])
            div_peak_threshold = max(abs_floor, adaptive_floor)
        else:
            div_peak_threshold = abs_floor
        div_peaks = _find_divergence_peaks_at(
            div_smooth, div_peak_threshold, cfg.candidate_max_count,
            nms_radius=cfg.div_peak_nms_radius,
            search_scale=cfg.div_peak_search_scale,
        )
        # Impact-event fast path: water-droplet hits produce a very strong
        # single-frame divergence spike.  Promote those peaks to "stable" on
        # the first frame they appear instead of waiting tracker_min_age,
        # so the heatmap responds the instant the droplet lands.
        if cfg.impact_event_enable:
            impact_val = div_peak_threshold * cfg.impact_div_multiplier
        else:
            impact_val = float("inf")
        self._tracked_centers, stable_peaks = _update_tracked_centers(
            div_peaks, self._tracked_centers,
            cfg.tracker_max_dist, cfg.tracker_min_age,
            impact_val=impact_val,
        )
        candidates = stable_peaks  # [(cx, cy, div_val)]

        # ---- Score each candidate ------------------------------------
        # We keep ``scored`` as the public (cx, cy, structure) list because
        # downstream consumers (rendering, ActivityResult.candidates) expect
        # it.  In parallel we collect ``strengths`` — structure × confidence
        # in the candidate's divergence amplitude — which feeds the frame
        # energy calculation.  This is what differentiates "weak ripple"
        # from "big splash": same structure, very different div magnitude.
        scored: List[Tuple[int, int, float]] = []
        strengths: List[float] = []
        edges_for_scoring = (edges * (soft_fg > 0.0).astype(np.uint8)).astype(np.uint8)
        for cx, cy, div_val in candidates:
            spatial = _radius_bin_peak_edge_score(edges_for_scoring, gx, gy, cx, cy, cfg)
            radial_flow = _radial_expansion_score(flow, cx, cy, cfg)
            div_ring = _ring_pool_mean(div_pos, cx, cy, cfg.r_min, cfg.r_max)
            div_norm = float(_soft_saturate(np.asarray(div_ring, dtype=np.float32),
                                            divergence_cap))
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
            # Confidence from divergence magnitude — soft-saturate so a 16×
            # impact spike doesn't dominate entirely but is clearly stronger
            # than a 2× background ripple.  div_val is in the same units as
            # div_smooth (a peak value of the smoothed positive divergence).
            div_conf = float(div_val) / (float(div_val) + cfg.energy_div_softcap)
            strengths.append(structure * div_conf)

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

        # ---- Sliding windows (vectorised ring buffer + running sum) --
        def _push(scale: str) -> np.ndarray:
            sw = self._sw[scale]
            buf = sw["buf"]
            i = sw["idx"]
            if sw["count"] < sw["len"]:
                # Buffer not yet full: just add into the running sum.
                sw["sum"] += raw_score
                sw["count"] += 1
            else:
                # Buffer full: subtract the slot we're about to overwrite,
                # then add the new value.
                sw["sum"] += raw_score - buf[i]
            buf[i] = raw_score
            sw["idx"] = (i + 1) % sw["len"]
            return (sw["sum"] / max(sw["count"], 1)).astype(np.float32)

        short_s = _push("short")
        mid_s = _push("mid")
        long_s = _push("long")

        # ---- Short pulse: max over last N raw_score frames ------------
        # A 1–2-frame impact spike gets washed out by short_window=30 averaging,
        # but max-pooling the last few frames keeps it visible on the heatmap
        # for the human eye to catch.
        if self._pulse_buf is not None and self._pulse_buf.shape[0] > 0:
            self._pulse_buf[self._pulse_idx] = raw_score
            self._pulse_idx = (self._pulse_idx + 1) % self._pulse_buf.shape[0]
            self._pulse_count = min(self._pulse_count + 1, self._pulse_buf.shape[0])
            pulse_grid = self._pulse_buf[: self._pulse_count].max(axis=0)
        else:
            pulse_grid = raw_score

        # Final heatmap: bilinear-upsampled mid_score (long-term context),
        # blended with the structure heatmap (per-candidate sharpness) and
        # the short pulse (instant impact responsiveness).
        mid_full = cv2.resize(mid_s, (W, H), interpolation=cv2.INTER_LINEAR)
        pulse_full = cv2.resize(pulse_grid, (W, H), interpolation=cv2.INTER_LINEAR)
        # Smooth structure heatmap to blend nicely with the bilinear fields
        struct_smooth = cv2.GaussianBlur(structure_hm, (0, 0), cfg.candidate_sigma * 0.6)
        w_pulse = float(cfg.short_pulse_weight)
        w_mid = 0.6 * (1.0 - w_pulse)
        w_struct = 0.4 * (1.0 - w_pulse)
        heatmap = np.clip(
            w_mid * mid_full + w_struct * struct_smooth + w_pulse * pulse_full,
            0.0, 1.0,
        ).astype(np.float32)

        # ---- Frame-level energy (probabilistic OR + envelope) --------
        # ``strengths`` is structure × divergence-confidence per candidate,
        # so the aggregation reflects both ripple quality and amplitude.
        energy_raw = _aggregate_candidate_energy(
            strengths, cfg.energy_candidate_weight
        )
        # Asymmetric EMA: attack fast (real onset visible right away),
        # release slow (smoothly fades through single-frame detection
        # misses).  Result: a bell-curve envelope that matches the
        # physical intuition of a ripple — fast rise, slow decay.
        a_atk = float(np.clip(cfg.energy_attack_alpha, 0.0, 1.0))
        a_rel = float(np.clip(cfg.energy_release_alpha, 0.0, 1.0))
        if a_atk <= 0.0 and a_rel <= 0.0:
            # Legacy single-rate fallback
            alpha = float(np.clip(cfg.energy_ema_alpha, 0.0, 1.0))
        else:
            alpha = a_atk if energy_raw > self._energy_ema else a_rel
        self._energy_ema = alpha * energy_raw + (1.0 - alpha) * self._energy_ema
        energy_smooth = float(np.clip(self._energy_ema, 0.0, 1.0))

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
            energy=energy_smooth,
            energy_raw=float(energy_raw),
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

            # Use the new scalar energy (0-1 ripple intensity of THIS frame)
            # as the headline number — it tracks the actual ripple activity
            # far better than mid_score.max(), which is a slow grid-pooled
            # rolling mean that lags 30+ frames behind a real event.
            energy = float(result.energy)
            energy_raw = float(result.energy_raw)
            label = activity_level(energy)
            text = (f"{label}  energy={energy:.2f}  "
                    f"raw={energy_raw:.2f}  "
                    f"f={result.frame_index}  cand={len(result.candidates)}")
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
                    f"  frame {n}{tot}  energy={energy:.2f}  "
                    f"raw={energy_raw:.2f}  level={label:<7}  "
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
