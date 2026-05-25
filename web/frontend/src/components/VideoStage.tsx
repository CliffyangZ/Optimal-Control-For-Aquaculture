import type { RefObject } from 'react';
import type { DetectionPayload } from '../types/monitoring';

function colorForClass(className: string) {
  let hash = 0;
  for (const char of className) hash = (hash * 31 + char.charCodeAt(0)) % 360;
  return `hsl(${hash} 90% 58%)`;
}

export function VideoStage({ videoRef, detections }: { videoRef: RefObject<HTMLVideoElement>; detections: DetectionPayload[] }) {
  return (
    <div className="video-stage">
      <video ref={videoRef} autoPlay playsInline muted className="live-video" />
      <div className="overlay-layer" aria-label="Detection overlays">
        {detections.map((detection, index) => {
          const color = colorForClass(detection.class_name);
          return (
            <div
              key={`${detection.class_name}-${index}-${detection.x}-${detection.y}`}
              className="bbox"
              style={{
                left: `${detection.x * 100}%`,
                top: `${detection.y * 100}%`,
                width: `${detection.width * 100}%`,
                height: `${detection.height * 100}%`,
                borderColor: color,
              }}
            >
              <span style={{ backgroundColor: color }}>{detection.class_name} {(detection.confidence * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
