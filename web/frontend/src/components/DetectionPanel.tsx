import type { OverlayMessage } from '../types/streaming'

interface DetectionPanelProps {
  overlay: OverlayMessage
  staleMs: number
}

export function DetectionPanel({ overlay, staleMs }: DetectionPanelProps) {
  const isStale = Date.now() - overlay.timestamp_ms > staleMs
  const { activity, detection_status, message } = overlay

  if (isStale) {
    return (
      <div className="detection-panel">
        <p className="detection-panel__title">Detection</p>
        <p className="status-message status-message--warn">Stale detection data</p>
      </div>
    )
  }

  if (detection_status === 'unavailable' || detection_status === 'degraded') {
    return (
      <div className="detection-panel">
        <p className="detection-panel__title">Detection</p>
        <p className="status-message status-message--warn">
          {message ?? `Detection ${detection_status}`}
        </p>
      </div>
    )
  }

  if (!activity) return null

  const level = activity.level.toLowerCase()

  return (
    <div className="detection-panel">
      <p className="detection-panel__title">Activity</p>
      <p className={`detection-panel__badge detection-panel__badge--${level}`}>
        {activity.level}
      </p>
      <p className="detection-panel__score">
        Score: {activity.score}/100 &nbsp; Confidence: {(activity.confidence * 100).toFixed(0)}%
      </p>
      {detection_status === 'delayed' && (
        <p className="status-message status-message--warn">Detection delayed</p>
      )}
    </div>
  )
}
