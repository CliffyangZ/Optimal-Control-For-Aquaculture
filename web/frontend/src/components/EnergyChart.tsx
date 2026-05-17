import { useEffect, useMemo, useRef, useState } from 'react'
import type { OverlayMessage } from '../types/streaming'

/**
 * Real-time ripple energy time-series chart.
 *
 * Subscribes to the latest overlay message and appends one sample per new
 * frame_index (drops duplicates if the same overlay is re-emitted). The
 * chart is drawn on an HTML5 canvas so it stays fast even when the detection
 * worker pushes overlays at 10+ Hz; we only repaint when a new sample lands
 * or when the canvas is resized.
 *
 * Two series are rendered:
 *   - energy_raw  → translucent line — the per-frame raw aggregation
 *   - energy      → bold line + filled area — the EMA-smoothed envelope
 *
 * The y-axis is fixed to [0, 1] (the energy is already normalized by the
 * detector). The x-axis is the latest `maxSamples` samples — older ones
 * scroll off the left edge.
 */

interface EnergySample {
  /** Detector-internal monotonic frame counter. Used to dedupe duplicate WS pushes. */
  frame_index: number
  /** Smoothed energy in [0, 1]. */
  energy: number
  /** Raw per-frame energy in [0, 1]. */
  energy_raw: number
  /** Browser wall-clock time (ms) when the sample was received. */
  timestamp_ms: number
}

interface EnergyChartProps {
  /** Latest overlay message; sampled on each render. */
  overlay: OverlayMessage | null
  /** Drop overlays older than this many ms (mirrors the OverlayLayer behavior). */
  staleMs: number
  /** Hard cap on the number of samples kept in memory / rendered. */
  maxSamples?: number
  /** When toggled (e.g. on session start), the history is cleared. */
  resetKey?: number | string
}

const DEFAULT_MAX_SAMPLES = 240 // ~24s at 10 fps detection target

export function EnergyChart({
  overlay,
  staleMs,
  maxSamples = DEFAULT_MAX_SAMPLES,
  resetKey,
}: EnergyChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  // Ring buffer of recent samples. We mutate it in place (no setState on the
  // hot path) because the canvas redraws are explicit and we don't need React
  // to re-render on every new frame — only when the headline text changes.
  const samplesRef = useRef<EnergySample[]>([])
  const lastFrameIndexRef = useRef<number>(-1)
  // Tick state is bumped on each new sample so the readout text below the
  // chart stays in sync with the latest values. The canvas itself is drawn
  // imperatively from a ref (not React state) to keep the hot path cheap.
  const [tick, setTick] = useState(0)
  const forceRender = () => setTick((t) => t + 1)

  // Reset history when the parent signals (e.g. session start/stop).
  useEffect(() => {
    samplesRef.current = []
    lastFrameIndexRef.current = -1
    forceRender()
    drawChart(canvasRef.current, samplesRef.current, maxSamples)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetKey, maxSamples])

  // Ingest the latest overlay → append a new sample if it's a new frame.
  useEffect(() => {
    if (!overlay) return
    const now = Date.now()
    if (now - overlay.timestamp_ms > staleMs) return
    const activity = overlay.activity
    if (!activity) return

    // frame_index is the canonical de-dupe key when present. Fall back to
    // the overlay's wall-clock timestamp when the backend omits it.
    const frameIndex =
      typeof activity.frame_index === 'number'
        ? activity.frame_index
        : overlay.timestamp_ms
    if (frameIndex <= lastFrameIndexRef.current) return
    lastFrameIndexRef.current = frameIndex

    const energy = clamp01(activity.energy ?? activity.score / 100)
    const energyRaw = clamp01(activity.energy_raw ?? energy)
    const sample: EnergySample = {
      frame_index: frameIndex,
      energy,
      energy_raw: energyRaw,
      timestamp_ms: overlay.timestamp_ms,
    }
    samplesRef.current.push(sample)
    if (samplesRef.current.length > maxSamples) {
      samplesRef.current.splice(0, samplesRef.current.length - maxSamples)
    }
    drawChart(canvasRef.current, samplesRef.current, maxSamples)
    forceRender()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay, staleMs, maxSamples])

  // Repaint on canvas resize so the chart stays sharp. ResizeObserver is
  // missing in jsdom (used by vitest); fall back to a single initial paint
  // in that environment so unit tests don't blow up.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawChart(canvas, samplesRef.current, maxSamples)
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => {
      drawChart(canvas, samplesRef.current, maxSamples)
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [maxSamples])

  // `tick` bumps on every new sample → recompute the readout text.
  const headline = useMemo(() => {
    const latest = samplesRef.current[samplesRef.current.length - 1]
    if (!latest) return { energy: '—', raw: '—', label: 'no-data' as const }
    return {
      energy: latest.energy.toFixed(2),
      raw: latest.energy_raw.toFixed(2),
      label: labelForEnergy(latest.energy),
    }
  }, [tick])

  return (
    <div className="energy-chart">
      <div className="energy-chart__header">
        <span className="energy-chart__title">Energy</span>
        <span className={`energy-chart__badge energy-chart__badge--${headline.label}`}>
          {headline.label}
        </span>
      </div>
      <canvas ref={canvasRef} className="energy-chart__canvas" />
      <div className="energy-chart__readout">
        <span>
          smoothed <strong>{headline.energy}</strong>
        </span>
        <span>
          raw <strong>{headline.raw}</strong>
        </span>
        <span className="energy-chart__samples">
          {samplesRef.current.length}/{maxSamples} samples
        </span>
      </div>
    </div>
  )
}

// ---------- helpers ----------

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0
  if (v < 0) return 0
  if (v > 1) return 1
  return v
}

function labelForEnergy(e: number): 'idle' | 'low' | 'medium' | 'high' | 'intense' {
  // Mirrors core.ripple_algo.activity_level() so the badge color matches the
  // backend's classification (which the OverlayMessage already collapses to
  // Low/Medium/High for the existing badge; here we keep the 5-level scheme
  // for the chart's own legend).
  if (e < 0.15) return 'idle'
  if (e < 0.3) return 'low'
  if (e < 0.5) return 'medium'
  if (e < 0.7) return 'high'
  return 'intense'
}

function drawChart(
  canvas: HTMLCanvasElement | null,
  samples: readonly EnergySample[],
  maxSamples: number,
): void {
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  const cssWidth = canvas.clientWidth
  const cssHeight = canvas.clientHeight
  if (cssWidth === 0 || cssHeight === 0) return

  // Resize backing store to match device pixels for crisp lines.
  const targetW = Math.max(1, Math.floor(cssWidth * dpr))
  const targetH = Math.max(1, Math.floor(cssHeight * dpr))
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW
    canvas.height = targetH
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, cssWidth, cssHeight)

  // Background panel.
  ctx.fillStyle = '#0f1117'
  ctx.fillRect(0, 0, cssWidth, cssHeight)

  // Padding for axes / labels.
  const padL = 26
  const padR = 6
  const padT = 6
  const padB = 14
  const plotW = Math.max(1, cssWidth - padL - padR)
  const plotH = Math.max(1, cssHeight - padT - padB)

  // Y gridlines + labels at 0, 0.25, 0.5, 0.75, 1.0.
  ctx.strokeStyle = '#2a2d3a'
  ctx.fillStyle = '#94a3b8'
  ctx.lineWidth = 1
  ctx.font = '10px system-ui, -apple-system, sans-serif'
  ctx.textBaseline = 'middle'
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  ticks.forEach((t) => {
    const y = padT + (1 - t) * plotH
    ctx.beginPath()
    ctx.moveTo(padL, y)
    ctx.lineTo(padL + plotW, y)
    ctx.stroke()
    ctx.textAlign = 'right'
    ctx.fillText(t.toFixed(2), padL - 4, y)
  })

  // X axis baseline.
  ctx.strokeStyle = '#475569'
  ctx.beginPath()
  ctx.moveTo(padL, padT + plotH)
  ctx.lineTo(padL + plotW, padT + plotH)
  ctx.stroke()

  if (samples.length === 0) {
    ctx.fillStyle = '#475569'
    ctx.textAlign = 'center'
    ctx.fillText(
      'Start monitoring to stream live energy…',
      padL + plotW / 2,
      padT + plotH / 2,
    )
    return
  }

  // X mapping: align the right edge with the most recent sample, even before
  // the buffer is full, so the line "fills" the chart left-to-right naturally.
  const denom = Math.max(1, maxSamples - 1)
  const xForIdxInBuffer = (idxFromRight: number): number => {
    const x = plotW - (idxFromRight / denom) * plotW
    return padL + x
  }

  const n = samples.length
  const lastIdx = n - 1

  // Raw line (thin / translucent).
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)'
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    const s = samples[i]
    const idxFromRight = lastIdx - i
    const x = xForIdxInBuffer(idxFromRight)
    const y = padT + (1 - s.energy_raw) * plotH
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Smoothed line + filled area under it.
  // 1) build path
  const pathX: number[] = new Array(n)
  const pathY: number[] = new Array(n)
  for (let i = 0; i < n; i++) {
    const s = samples[i]
    const idxFromRight = lastIdx - i
    pathX[i] = xForIdxInBuffer(idxFromRight)
    pathY[i] = padT + (1 - s.energy) * plotH
  }
  // 2) area fill
  const gradient = ctx.createLinearGradient(0, padT, 0, padT + plotH)
  gradient.addColorStop(0, 'rgba(74, 222, 128, 0.45)')
  gradient.addColorStop(1, 'rgba(74, 222, 128, 0.02)')
  ctx.fillStyle = gradient
  ctx.beginPath()
  ctx.moveTo(pathX[0], padT + plotH)
  for (let i = 0; i < n; i++) ctx.lineTo(pathX[i], pathY[i])
  ctx.lineTo(pathX[n - 1], padT + plotH)
  ctx.closePath()
  ctx.fill()
  // 3) line on top
  ctx.lineWidth = 1.5
  ctx.strokeStyle = '#4ade80'
  ctx.beginPath()
  for (let i = 0; i < n; i++) {
    if (i === 0) ctx.moveTo(pathX[i], pathY[i])
    else ctx.lineTo(pathX[i], pathY[i])
  }
  ctx.stroke()

  // Current-value marker.
  const last = samples[n - 1]
  const cx = pathX[n - 1]
  const cy = pathY[n - 1]
  ctx.fillStyle = '#4ade80'
  ctx.beginPath()
  ctx.arc(cx, cy, 2.5, 0, Math.PI * 2)
  ctx.fill()

  // Time-window label.
  ctx.fillStyle = '#94a3b8'
  ctx.textAlign = 'right'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(`f=${last.frame_index}`, padL + plotW, padT + plotH + 11)
  ctx.textAlign = 'left'
  ctx.fillText(`${maxSamples} samples`, padL, padT + plotH + 11)
}


