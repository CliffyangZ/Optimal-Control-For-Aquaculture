# Frontend Feed Analysis Integration Guide

## WebSocket Message Format (v2)

The backend now sends enriched detection messages in schema `detections.v2`:

```typescript
interface DetectionsMessage {
  schema: "detections.v2";
  source_id: string;
  frame_id: number;
  timestamp_ms: number;
  source_width: number;
  source_height: number;
  stream_status: string;
  detection_status: string;
  
  // Original YOLO detections (for backward compatibility)
  detections: DetectionPayload[];
  
  // NEW: Multi-detector results
  detectors: DetectorResult[];
  
  // NEW: Current frame feed area metrics
  feed_area: {
    raw_area: number;       // Pixel area of feed detections
    kalman_area: number;    // Kalman-smoothed area estimate
    detection_count: number; // Number of feed detections
  };
  
  // NEW: Historical data for charting
  feed_analysis: {
    frame_ids: number[];     // X-axis: frame numbers
    raw_areas: number[];     // Y-axis: raw measurements
    kalman_areas: number[];  // Y-axis: smoothed estimates
  };
}

interface DetectorResult {
  model_name: "yolo" | "rfdetr";
  inference_ms: number;
  detections: DetectionPayload[];
}

interface DetectionPayload {
  class_name: "fish" | "feed" | "debris" | "lumpfish";
  confidence: number;     // [0.0, 1.0]
  x: number;             // [0.0, 1.0] normalized
  y: number;             // [0.0, 1.0] normalized
  width: number;         // [0.0, 1.0] normalized
  height: number;        // [0.0, 1.0] normalized
}
```

## Display Feed Analysis Chart

### Setup Chart Library

```typescript
// Use Chart.js or similar
import Chart from 'chart.js/auto';

const feedAnalysisChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],  // frame_ids
    datasets: [
      {
        label: 'Raw Feed Area',
        data: [],
        borderColor: 'rgb(255, 99, 132)',  // Red
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        tension: 0.1,
        borderWidth: 1,
        pointRadius: 0,
      },
      {
        label: 'Kalman Filter (Smoothed)',
        data: [],
        borderColor: 'rgb(75, 192, 192)',  // Teal
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Feed Area Analysis (10s window)',
      },
      legend: {
        display: true,
        position: 'top',
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Area (pixels)',
        },
        beginAtZero: true,
      },
      x: {
        title: {
          display: true,
          text: 'Frame Number',
        },
      },
    },
  },
});
```

### Update Chart with WebSocket Data

```typescript
websocket.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.schema !== 'detections.v2') {
    return; // Handle other message types
  }

  // Update chart data
  if (message.feed_analysis) {
    feedAnalysisChart.data.labels = message.feed_analysis.frame_ids.map(
      (id) => `#${id}`
    );
    feedAnalysisChart.data.datasets[0].data =
      message.feed_analysis.raw_areas;
    feedAnalysisChart.data.datasets[1].data =
      message.feed_analysis.kalman_areas;
    feedAnalysisChart.update('none'); // No animation for smooth updates
  }

  // Update metrics display
  if (message.feed_area) {
    document.getElementById('raw-area').textContent =
      `${message.feed_area.raw_area.toFixed(0)} px²`;
    document.getElementById('kalman-area').textContent =
      `${message.feed_area.kalman_area.toFixed(0)} px²`;
    document.getElementById('detection-count').textContent =
      message.feed_area.detection_count;

    // Calculate smoothing ratio
    const ratio = message.feed_area.kalman_area > 0
      ? (1 - Math.abs(message.feed_area.kalman_area - message.feed_area.raw_area) / message.feed_area.kalman_area * 100).toFixed(1)
      : 0;
    document.getElementById('smoothing-ratio').textContent = `${ratio}%`;
  }

  // Update detections based on selected detector
  const selectedDetector = getSelectedDetector(); // "yolo" or "rfdetr"
  const detectorResult = message.detectors.find(
    (d) => d.model_name === selectedDetector
  );

  if (detectorResult) {
    renderDetections(
      message.source_width,
      message.source_height,
      detectorResult.detections,
      detectorResult.model_name
    );
    document.getElementById('inference-time').textContent =
      `${detectorResult.inference_ms.toFixed(1)}ms`;
  }
};
```

## HTML Layout Example

```html
<div class="detection-panel">
  <!-- Detector Selection -->
  <div class="detector-controls">
    <button id="detector-yolo" class="detector-btn active" data-detector="yolo">
      YOLO Results
    </button>
    <button id="detector-rfdetr" class="detector-btn" data-detector="rfdetr">
      RF-DETR Results
    </button>
    <span id="inference-time" class="inference-time">-ms</span>
  </div>

  <!-- Metrics Display -->
  <div class="feed-metrics">
    <div class="metric">
      <label>Raw Feed Area</label>
      <span id="raw-area" class="value">-</span> px²
    </div>
    <div class="metric">
      <label>Kalman Filtered</label>
      <span id="kalman-area" class="value">-</span> px²
    </div>
    <div class="metric">
      <label>Feed Detections</label>
      <span id="detection-count" class="value">-</span>
    </div>
    <div class="metric">
      <label>Smoothing Ratio</label>
      <span id="smoothing-ratio" class="value">-</span> %
    </div>
  </div>

  <!-- Feed Analysis Chart -->
  <div class="chart-container">
    <canvas id="feed-analysis-chart"></canvas>
  </div>

  <!-- Detection Visualization -->
  <div class="detection-canvas-container">
    <canvas id="detection-canvas" width="640" height="480"></canvas>
  </div>
</div>
```

## CSS Styling Example

```css
.detection-panel {
  display: grid;
  grid-template-rows: auto auto 1fr auto;
  gap: 12px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 8px;
}

.detector-controls {
  display: flex;
  gap: 8px;
  align-items: center;
}

.detector-btn {
  padding: 8px 12px;
  border: 2px solid #ddd;
  background: white;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
}

.detector-btn.active {
  border-color: #2196f3;
  background: #2196f3;
  color: white;
}

.inference-time {
  margin-left: auto;
  font-size: 12px;
  color: #666;
  font-family: monospace;
}

.feed-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}

.metric {
  display: flex;
  flex-direction: column;
  padding: 8px;
  background: white;
  border-radius: 4px;
  border-left: 3px solid #2196f3;
}

.metric label {
  font-size: 12px;
  color: #999;
  font-weight: 600;
  text-transform: uppercase;
}

.metric .value {
  font-size: 18px;
  font-weight: bold;
  color: #333;
  margin-top: 4px;
  font-family: monospace;
}

.chart-container {
  position: relative;
  height: 300px;
  background: white;
  border-radius: 4px;
  padding: 8px;
}

.detection-canvas-container {
  position: relative;
  background: white;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
}

canvas#detection-canvas {
  max-width: 100%;
  height: auto;
}
```

## Detector Switching Logic

```typescript
let currentDetector: "yolo" | "rfdetr" = "yolo";

document.querySelectorAll(".detector-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".detector-btn").forEach((b) => {
      b.classList.remove("active");
    });
    btn.classList.add("active");

    const detector = btn.getAttribute("data-detector") as
      | "yolo"
      | "rfdetr";
    if (detector && currentDetector !== detector) {
      currentDetector = detector;
      // Chart and detections will auto-update on next WebSocket message
    }
  });
});

function getSelectedDetector() {
  return currentDetector;
}
```

## Performance Considerations

1. **Chart Updates**: Use `update('none')` to skip animations
2. **History Size**: 300 frames = ~10s at 30fps
3. **Canvas Rendering**: Debounce if frame rate is >30fps
4. **Memory**: History automatically pruned by backend

## Backward Compatibility

- Old schema `detections.v1` still supported
- `detections` field maintained in v2 for backward compatibility
- Frontend can check `message.schema` to handle both versions:

```typescript
if (message.schema === "detections.v2") {
  // Use new feed_analysis and detectors fields
} else {
  // Use legacy detections field
}
```

## Troubleshooting

### Chart not updating

1. Verify WebSocket connection is active
2. Check browser console for errors
3. Ensure `feed_analysis` data is present in message

### Feed area always zero

1. Check YOLO confidence threshold (may be filtering all detections)
2. Verify feed training data quality
3. Inspect raw detections in browser DevTools

### RF-DETR not showing up

1. Verify `RFDETR_MODEL_PATH` is set on backend
2. Check backend logs for loading errors
3. Detector list may be empty if model failed to load

## Next Steps

- [ ] Implement feed consumption rate calculation
- [ ] Add multi-zone area analysis
- [ ] Implement anomaly detection alerts
- [ ] Add time-series export (CSV/JSON)
- [ ] Real-time statistics panel (mean, std, min, max)
