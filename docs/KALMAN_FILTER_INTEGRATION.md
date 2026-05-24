# Kalman Filter Integration Guide

## Overview

The Kalman Filter module (`core.infer.kalman_filter`) provides robust state estimation for feed area measurements, automatically handling:
- **Zero values** (detection failures) — skipped automatically
- **Outliers** (>3σ deviations) — optional rejection
- **Noise reduction** — smooths noisy detector outputs
- **RTS smoothing** — optional post-hoc bias reduction using full trajectory

## Quick Start

```python
from core.infer import create_kalman_filter, KalmanFilterConfig

# Create filter with recommended config
config = KalmanFilterConfig(
    sigma_R=0.15,      # Measurement noise
    sigma_Q=5e-3,      # Process noise
    reject_outliers=True,
    outlier_threshold=3.0,
)
kf = create_kalman_filter(config)

# Filter measurements
for area_measurement in feed_areas:
    filtered_area = kf.update(
        area_measurement,
        reject_outliers=config.reject_outliers,
        outlier_threshold=config.outlier_threshold,
    )
    print(f"Raw: {area_measurement:,.0f} → Filtered: {filtered_area:,.0f}")

# Optional: Smooth entire trajectory using backward information
smoothed_areas = kf.rts_smooth()
```

## Configuration

### Default Config
```python
KalmanFilterConfig(
    sigma_R=0.15,          # Measurement noise std dev
    sigma_Q=5e-3,          # Process noise std dev
    reject_outliers=True,  # Enable outlier rejection
    outlier_threshold=3.0, # Reject if >3σ from prediction
)
```

### Config Tuning

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `sigma_R` | Higher = trust measurements less | 0.1–0.3 |
| `sigma_Q` | Higher = allow faster state changes | 1e-4 to 1e-2 |
| `outlier_threshold` | Higher = reject fewer outliers | 2.0–4.0 |

**When to adjust:**
- **High sensor noise** → increase `sigma_R` (e.g., 0.2–0.3)
- **Detector misses feed suddenly** → increase `sigma_Q` (e.g., 1e-2)
- **Too many spikes** → decrease `outlier_threshold` (e.g., 2.5)

## Integration with Detector

### Approach 1: Post-Processing
Apply Kalman Filter after collecting all detections from a video:

```python
from core.infer import create_kalman_filter, KalmanFilterConfig
from core.infer.rfdetr_detector import RFDETRDetector

detector = RFDETRDetector.load("path/to/model.pt")
config = KalmanFilterConfig()
kf = create_kalman_filter(config)

video = cv2.VideoCapture("video.mp4")
frame_id = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    detection_result = detector.detect(frame, frame_id)

    # Extract feed area if detected
    feed_area = 0.0
    for det in detection_result.detections:
        if det.class_name == "feed":
            feed_area = det.bbox_w * det.bbox_h  # Normalized area
            break

    # Filter
    filtered_area = kf.update(feed_area, reject_outliers=True)
    frame_id += 1

# Smooth using RTS
smoothed = kf.rts_smooth()
```

### Approach 2: Real-Time Streaming
Use Kalman Filter during live streaming (no backward smoothing):

```python
class FilteredDetector:
    def __init__(self, model_path: str, config=None):
        self.detector = RFDETRDetector.load(model_path)
        self.kf = create_kalman_filter(config or KalmanFilterConfig())

    def detect_and_filter(self, frame, frame_id):
        result = self.detector.detect(frame, frame_id)

        feed_area = 0.0
        for det in result.detections:
            if det.class_name == "feed":
                feed_area = det.bbox_w * det.bbox_h
                break

        filtered_area = self.kf.update(
            feed_area,
            reject_outliers=True,
        )

        # Return result with filtered area
        return filtered_area
```

## Understanding the Filters

### LinearKalmanFilter
Basic 1D filter for any continuous measurement:
```python
from core.infer import KalmanFilter

kf = KalmanFilter(sigma_R=0.1, sigma_Q=0.01)
estimate = kf.update(100.0)
```

### LogNormalKalmanFilter (Default)
Specialized for log-normal distributed data (like feed areas):
```python
from core.infer import LogNormalKalmanFilter

kf = LogNormalKalmanFilter(sigma_R=0.15, sigma_Q=5e-3)
estimate = kf.update(50000.0)  # Returns area in original space
```

**Why log-space?**
- Feed areas span 1–700k pixels² (7-order-of-magnitude range)
- Log-normal distribution captures this better than Gaussian
- Numeric stability: prevents underflow/overflow

## Advanced Features

### Zero-Value Handling
Measurements ≤ 0 are automatically skipped (treated as detection failures):
```python
kf.update(0.0)  # Returns previous estimate, no state change
```

### Outlier Rejection
Detections >3σ from prediction are rejected (keep previous estimate):
```python
kf.update(999999.0, reject_outliers=True, outlier_threshold=3.0)
```

### RTS Backward Smoothing
Post-hoc smoothing using entire trajectory (requires all data upfront):
```python
# After all forward updates
smoothed_estimates = kf.rts_smooth()
# Reduces variance, removes spikes from forward filter
```

## Example: Feed Consumption Estimation

```python
config = KalmanFilterConfig(
    sigma_R=0.15,
    sigma_Q=5e-3,
    reject_outliers=True,
)
kf = create_kalman_filter(config)

feed_areas = []
consumption_rates = []

for frame_id in range(n_frames):
    # Detect + filter
    result = detector.detect(frame, frame_id)
    raw_area = extract_feed_area(result)
    filtered_area = kf.update(raw_area, reject_outliers=True)
    feed_areas.append(filtered_area)

    # Estimate consumption rate
    if frame_id > 0:
        rate = (feed_areas[frame_id - 1] - filtered_area) / frame_id
        consumption_rates.append(rate)

# Smooth for final report
smoothed_areas = kf.rts_smooth()
```

## Testing & Validation

See `tests/test_kalman_filter.py` for comprehensive test suite:
```bash
python -m pytest tests/test_kalman_filter.py -v
```

Key test scenarios:
- Zero-value skipping
- Outlier rejection
- Sequential convergence
- RTS smoothing variance reduction
- Log-space numerical stability

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single update | <1 μs | Real-time safe |
| RTS smoothing | O(n) | 1M frames ≈ 1ms |
| History storage | O(n) | ~40 bytes/frame |

## Troubleshooting

### Spiky Output
**Problem:** Filtered curve still has spikes
**Solution:**
1. Increase `sigma_Q` → allows faster adaptation
2. Decrease `outlier_threshold` → reject more outliers
3. Add RTS smoothing → uses full trajectory information

### Over-Smoothing
**Problem:** Filter lags real changes
**Solution:**
1. Decrease `sigma_R` → trust measurements more
2. Increase `sigma_Q` → allow quicker state changes

### Extreme Values
**Problem:** Filtered values jump to unrealistic ranges
**Solution:**
1. Lower `outlier_threshold` (default 3.0 → try 2.5)
2. Increase `sigma_Q` to allow smoother transitions
3. Check raw detector output for systematic biases

## References

- **Kalman Filter Theory**: [Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
- **RTS Smoothing**: Rauch, C. T., Striebel, C. T., & Tung, F. (1965)
- **Implementation Details**: See `KF_SPIKE_FIX_GUIDE.md`
