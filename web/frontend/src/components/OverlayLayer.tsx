import { useEffect, useState } from 'react'
import type { OverlayMessage, OverlayRegion } from '../types/streaming'

interface OverlayLayerProps {
  overlay: OverlayMessage
  videoRef: React.RefObject<HTMLVideoElement | null>
  staleMs: number
}

export function OverlayLayer({ overlay, videoRef, staleMs }: OverlayLayerProps) {
  const [videoRect, setVideoRect] = useState<DOMRect | null>(null)

  // Track rendered video content rect (letterboxing aware)
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

  const isStale = Date.now() - overlay.timestamp_ms > staleMs
  if (isStale || !videoRect || overlay.regions.length === 0) return null

  return (
    <svg className="overlay-svg" viewBox={`0 0 ${videoRect.width} ${videoRect.height}`}
      style={{ left: videoRect.x, top: videoRect.y, width: videoRect.width, height: videoRect.height, position: 'absolute' }}>
      {overlay.regions.map((region) => (
        <OverlayRegionBox
          key={region.id}
          region={region}
          width={videoRect.width}
          height={videoRect.height}
        />
      ))}
    </svg>
  )
}

function OverlayRegionBox({
  region,
  width,
  height,
}: {
  region: OverlayRegion
  width: number
  height: number
}) {
  const x = region.x * width
  const y = region.y * height
  const w = region.width * width
  const h = region.height * height
  return (
    <g>
      <rect
        className="overlay-region"
        x={x}
        y={y}
        width={w}
        height={h}
      />
      <text className="overlay-region-label" x={x + 4} y={y - 4}>
        {region.id}
      </text>
    </g>
  )
}
