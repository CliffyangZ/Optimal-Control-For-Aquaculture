"""CLI for testing the ripple detection algorithm on a local video file.

Examples:
    uv run main.py                                  # default data/fish_ripple.mov, live preview
    uv run main.py data/simple_ripple.mov
    uv run main.py data/fish_ripple.mov --no-show --output out.mp4
    uv run main.py data/fish_ripple.mov --max-frames 300 --short-window 10
    uv run main.py data/fish_ripple.mov --input-scale 0.5 --flow-scale 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.ripple_algo import RippleAlgoConfig, run_on_video

DEFAULT_VIDEO = "data/fish_ripple.mov"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ripple detector — video test runner")
    p.add_argument("video", nargs="?", default=DEFAULT_VIDEO,
                   help=f"Path to input video (default: {DEFAULT_VIDEO})")
    p.add_argument("--output", "-o", default=None,
                   help="Save annotated output to this path (e.g. out.mp4)")
    p.add_argument("--no-show", action="store_true",
                   help="Disable live preview window (headless mode, e.g. WSL)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames")

    # Key config knobs exposed as CLI flags
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="Downscale input before processing (e.g. 0.5 for 1080p→540p)")
    p.add_argument("--flow-scale", type=float, default=0.5,
                   help="Optical flow downscale factor (default 0.5)")
    p.add_argument("--grid", type=int, default=8,
                   help="Grid size N → N×N cells (default 8)")
    p.add_argument("--short-window", type=int, default=30,
                   help="Short sliding-window length in frames (default 30 ≈ 1 s)")
    p.add_argument("--mid-window", type=int, default=150,
                   help="Mid sliding-window length in frames (default 150 ≈ 5 s)")
    p.add_argument("--long-window", type=int, default=900,
                   help="Long sliding-window length in frames (default 900 ≈ 30 s)")

    # Sensitivity presets
    p.add_argument("--sensitive", action="store_true",
                   help="Skip pre-blur and lower Canny thresholds to catch faint ripples")
    p.add_argument("--blur-kernel", type=int, default=None,
                   help="Pre-blur kernel (0 to skip). Default uses config (3).")
    p.add_argument("--canny-low", type=int, default=None,
                   help="Canny low threshold (default 15, sensitive: 10)")
    p.add_argument("--canny-high", type=int, default=None,
                   help="Canny high threshold (default 60, sensitive: 40)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Max candidate ripple centres per frame (default 16)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.video).is_file():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Sensitivity preset → fine-grained overrides take precedence
    if args.sensitive:
        sens_blur, sens_low, sens_high, sens_cand = 0, 10, 40, 32
    else:
        sens_blur = sens_low = sens_high = sens_cand = None

    cfg = RippleAlgoConfig(
        input_scale=args.input_scale,
        flow_scale=args.flow_scale,
        grid_rows=args.grid,
        grid_cols=args.grid,
        short_window=args.short_window,
        mid_window=args.mid_window,
        long_window=args.long_window,
    )
    if args.blur_kernel is not None:
        cfg.blur_kernel = args.blur_kernel
    elif sens_blur is not None:
        cfg.blur_kernel = sens_blur
    if args.canny_low is not None:
        cfg.canny_low = args.canny_low
    elif sens_low is not None:
        cfg.canny_low = sens_low
    if args.canny_high is not None:
        cfg.canny_high = args.canny_high
    elif sens_high is not None:
        cfg.canny_high = sens_high
    if args.max_candidates is not None:
        cfg.candidate_max_count = args.max_candidates
    elif sens_cand is not None:
        cfg.candidate_max_count = sens_cand

    print(f"Video       : {args.video}")
    print(f"Grid        : {cfg.grid_rows}×{cfg.grid_cols}")
    print(f"Windows     : short={cfg.short_window}  mid={cfg.mid_window}  long={cfg.long_window}")
    print(f"Input scale : {cfg.input_scale}   |   Flow scale: {cfg.flow_scale}")
    print(f"Preprocess  : blur={cfg.blur_kernel}  CLAHE-first={cfg.clahe_before_blur}")
    print(f"Canny       : low={cfg.canny_low}  high={cfg.canny_high}")
    print(f"Candidates  : max={cfg.candidate_max_count}  min_area={cfg.candidate_min_area}"
          f"{'   (sensitive preset)' if args.sensitive else ''}")
    if args.output:
        print(f"Output      : {args.output}")
    print(f"Preview     : {'off' if args.no_show else 'on'}")
    print("─" * 56)

    try:
        run_on_video(
            video_path=args.video,
            cfg=cfg,
            show=not args.no_show,
            output_path=args.output,
            max_frames=args.max_frames,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
