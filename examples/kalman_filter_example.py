#!/usr/bin/env python3
"""Example: Using Kalman Filter to smooth feed area measurements.

Demonstrates:
1. Basic filtering of noisy measurements
2. Handling detection failures (zero values)
3. Outlier rejection
4. RTS backward smoothing
"""

import math
from pathlib import Path

from core.infer import create_kalman_filter, KalmanFilterConfig


def generate_synthetic_feed_sequence():
    """Generate synthetic feed area measurements with noise and dropouts."""
    measurements = []

    # Stable phase: 500k pixels
    for _ in range(10):
        noise = 50000 * (2 * (hash(_) % 100) / 100.0 - 1)
        measurements.append(500000 + noise)

    # Consumption phase: decline from 500k to 100k
    for i in range(20):
        base = 500000 - (400000 * i / 20)
        noise = 50000 * (2 * (hash(i) % 100) / 100.0 - 1)
        measurements.append(max(0, base + noise))

    # Detection failure (spikes)
    measurements[12] = 0.0  # Miss
    measurements[15] = 4500000.0  # Outlier spike
    measurements[16] = 0.0  # Miss

    # Stable at low level: 100k pixels
    for i in range(10):
        noise = 20000 * (2 * (hash(20 + i) % 100) / 100.0 - 1)
        measurements.append(100000 + noise)

    return measurements


def analyze_filtering(raw_areas, filtered_areas, smoothed_areas=None):
    """Print statistics comparing raw, filtered, and optional smoothed data."""
    print("\n" + "=" * 70)
    print("FILTERING ANALYSIS")
    print("=" * 70)

    def compute_stats(data):
        if not data:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = math.sqrt(variance)

        return {
            "mean": mean,
            "std": std,
            "min": min(data),
            "max": max(data),
        }

    raw_stats = compute_stats(raw_areas)
    filtered_stats = compute_stats(filtered_areas)

    print(f"\n{'Metric':<20} {'Raw':<15} {'Filtered':<15} {'Improvement':<15}")
    print("-" * 65)

    for key in ["mean", "std", "min", "max"]:
        raw_val = raw_stats[key]
        filt_val = filtered_stats[key]

        if key == "std":
            improvement = (raw_val - filt_val) / raw_val * 100
            print(f"{key:<20} {raw_val:>13,.0f}  {filt_val:>13,.0f}  {improvement:>12.1f}%")
        else:
            print(f"{key:<20} {raw_val:>13,.0f}  {filt_val:>13,.0f}")

    if smoothed_areas:
        smoothed_stats = compute_stats(smoothed_areas)
        smoothed_std = smoothed_stats["std"]
        improvement_vs_raw = (raw_stats["std"] - smoothed_std) / raw_stats["std"] * 100
        print(f"\n{'RTS Smoothed std':<20} {smoothed_std:>13,.0f}  (vs raw: {improvement_vs_raw:+.1f}%)")


def main():
    """Main example: filter synthetic feed area sequence."""
    print("\n" + "=" * 70)
    print("KALMAN FILTER EXAMPLE: Feed Area Smoothing")
    print("=" * 70)

    measurements = generate_synthetic_feed_sequence()
    print(f"\nGenerated {len(measurements)} measurements")
    print(f"Raw area range: {min([m for m in measurements if m > 0]):,.0f} to {max(measurements):,.0f} pixels²")

    config = KalmanFilterConfig(
        sigma_R=0.15,
        sigma_Q=5e-3,
        reject_outliers=True,
        outlier_threshold=3.0,
    )

    print("\n" + "-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"Measurement noise (sigma_R): {config.sigma_R}")
    print(f"Process noise (sigma_Q):     {config.sigma_Q}")
    print(f"Reject outliers:             {config.reject_outliers}")
    print(f"Outlier threshold:           {config.outlier_threshold}σ")

    kf = create_kalman_filter(config)

    print("\n" + "-" * 70)
    print("FILTERING (forward pass)")
    print("-" * 70)
    print(f"{'Frame':<8} {'Raw Area':<15} {'Filtered':<15} {'Action':<20}")
    print("-" * 70)

    filtered_results = []
    for i, measurement in enumerate(measurements):
        prev_est = kf.x_estimate

        filtered = kf.update(
            measurement,
            reject_outliers=config.reject_outliers,
            outlier_threshold=config.outlier_threshold,
        )
        filtered_results.append(filtered)

        action = ""
        if measurement <= 0:
            action = "↷ Skipped zero"
        elif measurement > 4e6:
            action = "⚠ Outlier rejected"

        if action or i % 5 == 0:
            print(f"{i:<8} {measurement:>13,.0f}  {filtered:>13,.0f}  {action:<20}")

    print("\n" + "-" * 70)
    print("RTS BACKWARD SMOOTHING")
    print("-" * 70)

    smoothed_results = kf.rts_smooth()

    print(f"Smoothed {len(smoothed_results)} estimates using full trajectory")
    print("(backward pass uses future information to reduce bias)")

    analyze_filtering(measurements, filtered_results, smoothed_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThe Kalman Filter:")
    print("✓ Handles zero measurements (detection failures)")
    print("✓ Rejects outliers (spikes >3σ from prediction)")
    print("✓ Reduces measurement noise")
    print("✓ Supports RTS smoothing for post-hoc bias reduction")
    print("\nUse cases:")
    print("- Real-time: update() for each frame during streaming")
    print("- Batch: update() all frames, then rts_smooth() for final report")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
