import type { OverlayMessage } from '../types/streaming'

interface DetectionPanelProps {
  overlay: OverlayMessage
  staleMs: number
}

type EnergyLevel = 'idle' | 'low' | 'medium' | 'high' | 'intense'

function energyLevel(e: number): EnergyLevel {
  if (e < 0.15) return 'idle'
  if (e < 0.30) return 'low'
  if (e < 0.50) return 'medium'
  if (e < 0.70) return 'high'
  return 'intense'
}

const LEVEL_LABEL: Record<EnergyLevel, string> = {
  idle: 'Idle',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  intense: 'Intense',
}

export function DetectionPanel({ overlay, staleMs }: DetectionPanelProps) {
  const isStale = Date.now() - overlay.timestamp_ms > staleMs
  const { activity, detection_status, message } = overlay

  if (isStale) {
    return (
      <div className="detection-panel">
        <p className="detection-panel__title">Ripple Energy</p>
        <p className="status-message status-message--warn">Stale detection data</p>
      </div>
    )
  }

  if (detection_status === 'unavailable' || detection_status === 'degraded') {
    return (
      <div className="detection-panel">
        <p className="detection-panel__title">Ripple Energy</p>
        <p className="status-message status-message--warn">
          {message ?? `Detection ${detection_status}`}
        </p>
      </div>
    )
  }

  if (!activity) return null

  const energy = typeof activity.energy === 'number' ? activity.energy : activity.score / 100
  const energyRaw = typeof activity.energy_raw === 'number' ? activity.energy_raw : energy
  const level = energyLevel(energy)
  const barPct = Math.round(energy * 100)
  const isPulsing = energy >= 0.15

  return (
    <div className="detection-panel">
      <div className="detection-panel__header">
        <p className="detection-panel__title">Ripple Energy</p>
        <span className={`energy-meter__level-badge energy-meter__level-badge--${level}`}>
          {LEVEL_LABEL[level]}
        </span>
      </div>

      {/* Big energy number */}
      <div className={`energy-meter__value energy-meter__value--${level}`}>
        {energy.toFixed(2)}
        <span className="energy-meter__value-raw">
          &nbsp;/ {energyRaw.toFixed(2)} raw
        </span>
      </div>

      {/* Animated progress bar */}
      <div className="energy-meter__bar-track">
        <div
          className={`energy-meter__bar-fill energy-meter__bar-fill--${level}${isPulsing ? ' energy-meter__bar-fill--pulse' : ''}`}
          style={{ width: `${barPct}%` }}
        />
        {/* Threshold markers */}
        <div className="energy-meter__marker" style={{ left: '15%' }} />
        <div className="energy-meter__marker" style={{ left: '30%' }} />
        <div className="energy-meter__marker" style={{ left: '50%' }} />
        <div className="energy-meter__marker" style={{ left: '70%' }} />
      </div>

      {/* Pulse ring indicator */}
      {isPulsing && (
        <div className={`energy-meter__pulse-ring energy-meter__pulse-ring--${level}`} />
      )}

      {detection_status === 'delayed' && (
        <p className="status-message status-message--warn">Detection delayed</p>
      )}
    </div>
  )
}
