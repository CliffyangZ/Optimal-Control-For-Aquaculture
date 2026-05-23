import { useEffect, useState } from 'react'
import type { DetectionPayload, DetectionsMessage } from '../types/streaming'

interface OverlayLayerProps {
  overlay: DetectionsMessage
  videoRef: React.RefObject<HTMLVideoElement | null>
  staleMs: number
}

const CLASS_LABEL_COLOR: Record<DetectionPayload['class_name'], string> = {
  fish: '#38bdf8',
  lumpfish: '#22d3ee',
  feed: '#f59e0b',
  debris: '#f87171',
}

export function OverlayLayer({ overlay, videoRef, staleMs }: OverlayLayerProps) {
  const [videoRect, setVideoRect] = useState<DOMRect | null>(null)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const measure = () => {
      const containerW = video.clientWidth
      const containerH = video.clientHeight
      const videoW = video.videoWidth || overlay.source_width
      const videoH = video.videoHeight || overlay.source_height
      if (!videoW || !videoH) {
        setVideoRect(new DOMRect(0, 0, containerW, containerH))
        return
      }
      const containerAspect = containerW / containerH
      const videoAspect = videoW / videoH
      let w: number, h: number, x: number, y: number
      if (containerAspect > videoAspect) {
        h = containerH
        w = h * videoAspect
        x = (containerW - w) / 2
        y = 0
      } else {
        w = containerW
        h = w / videoAspect
        x = 0
        y = (containerH - h) / 2
      }
      setVideoRect(new DOMRect(x, y, w, h))
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(video)
    return () => observer.disconnect()
  }, [videoRef, overlay.source_width, overlay.source_height])

  const isStale = Date.now() - overlay.timestamp_ms > staleMs || overlay.detection_status === 'stale'
  if (isStale || !videoRect || overlay.detections.length === 0) return null

  return (
    <svg
      className="overlay-svg"
      viewBox={`0 0 ${videoRect.width} ${videoRect.height}`}
      style={{ left: videoRect.x, top: videoRect.y, width: videoRect.width, height: videoRect.height, position: 'absolute' }}
    >
      {overlay.detections.map((detection, index) => (
        <DetectionBox
          key={`${detection.class_name}-${index}-${detection.x}-${detection.y}`}
          detection={detection}
          width={videoRect.width}
          height={videoRect.height}
        />
      ))}
    </svg>
  )
}

function DetectionBox({ detection, width, height }: { detection: DetectionPayload; width: number; height: number }) {
  const x = detection.x * width
  const y = detection.y * height
  const w = detection.width * width
  const h = detection.height * height
  const color = CLASS_LABEL_COLOR[detection.class_name]
  const label = `${detection.class_name} ${Math.round(detection.confidence * 100)}%`

  return (
    <g>
      <rect className="overlay-detection" x={x} y={y} width={w} height={h} stroke={color} />
      <rect className="overlay-detection-label-bg" x={x} y={Math.max(0, y - 22)} width={Math.max(72, label.length * 7)} height={18} fill={color} />
      <text className="overlay-detection-label" x={x + 5} y={Math.max(13, y - 9)}>
        {label}
      </text>
    </g>
  )
}
