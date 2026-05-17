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
    <svg
      className="overlay-svg"
      viewBox={`0 0 ${videoRect.width} ${videoRect.height}`}
      style={{
        left: videoRect.x,
        top: videoRect.y,
        width: videoRect.width,
        height: videoRect.height,
        position: 'absolute',
      }}
    >
      {overlay.regions.map((region, idx) => (
        <RippleCenter
          key={region.id}
          region={region}
          width={videoRect.width}
          height={videoRect.height}
          ringDelay={idx * 0.35}
        />
      ))}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Per-candidate ripple-centre visualisation
// ---------------------------------------------------------------------------

function scoreColor(score: number): string {
  if (score < 0.25) return '#38bdf8'  // cyan  — low
  if (score < 0.50) return '#4ade80'  // green — medium-low
  if (score < 0.70) return '#fb923c'  // orange — medium-high
  return '#f87171'                     // red   — high
}

function RippleCenter({
  region,
  width,
  height,
  ringDelay,
}: {
  region: OverlayRegion
  width: number
  height: number
  ringDelay: number
}) {
  // Centre of the bounding box in SVG pixels
  const cx = (region.x + region.width / 2) * width
  const cy = (region.y + region.height / 2) * height

  const score = region.score ?? 0.3
  const color = scoreColor(score)

  // Ring size scales with score: higher score → larger max radius
  const baseR = 10 + score * 14          // 10–24 px at start
  const maxR  = 28 + score * 48          // 28–76 px at end
  const dur   = 1.8 - score * 0.6        // 1.2–1.8 s (stronger = faster)
  const durStr = `${dur.toFixed(2)}s`
  const delay1 = `${ringDelay.toFixed(2)}s`
  const delay2 = `${(ringDelay + dur * 0.5).toFixed(2)}s`

  return (
    <g>
      {/* Expanding ring — first wave */}
      <circle cx={cx} cy={cy} r={baseR} fill="none" stroke={color} strokeWidth="1.5" opacity="0">
        <animate attributeName="r"       from={baseR} to={maxR}  dur={durStr} begin={delay1} repeatCount="indefinite" />
        <animate attributeName="opacity" from="0.75"  to="0"     dur={durStr} begin={delay1} repeatCount="indefinite" />
      </circle>

      {/* Expanding ring — second wave (half-period offset) */}
      <circle cx={cx} cy={cy} r={baseR} fill="none" stroke={color} strokeWidth="1" opacity="0">
        <animate attributeName="r"       from={baseR} to={maxR}  dur={durStr} begin={delay2} repeatCount="indefinite" />
        <animate attributeName="opacity" from="0.45"  to="0"     dur={durStr} begin={delay2} repeatCount="indefinite" />
      </circle>

      {/* Soft glow halo */}
      <circle cx={cx} cy={cy} r={baseR * 0.9} fill={color} opacity={0.10 + score * 0.08} />

      {/* Centre dot — solid, size ∝ score */}
      <circle cx={cx} cy={cy} r={3 + score * 4} fill={color} opacity="0.95" />

      {/* Score label */}
      <text
        x={cx + (3 + score * 4) + 4}
        y={cy - (3 + score * 4) - 2}
        fill={color}
        fontSize="10"
        fontFamily="system-ui, monospace"
        fontWeight="600"
        opacity="0.9"
      >
        {score.toFixed(2)}
      </text>
    </g>
  )
}
