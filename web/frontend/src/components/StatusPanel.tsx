import type { StreamStatus, DetectionStatus } from '../types/streaming'
import type { WebRtcState } from '../hooks/useWebRtcPlayer'

interface StatusPanelProps {
  streamStatus: StreamStatus
  detectionStatus: DetectionStatus
  rtcState: WebRtcState
  rtcError: string | null
  activeViewerCount: number
}

const STREAM_LABELS: Record<StreamStatus, string> = {
  waiting: 'Waiting',
  connecting: 'Connecting…',
  live: 'Live',
  reconnecting: 'Reconnecting…',
  stream_unavailable: 'Stream Unavailable',
  degraded: 'Degraded',
  error: 'Error',
}

const DETECTION_LABELS: Record<DetectionStatus, string> = {
  loading: 'Loading',
  active: 'Active',
  stale: 'Stale',
  unavailable: 'Unavailable',
}

export function StatusPanel({
  streamStatus,
  detectionStatus,
  rtcState,
  rtcError,
  activeViewerCount,
}: StatusPanelProps) {
  return (
    <div className="status-panel">
      <div className="status-row">
        <span className="status-label">Stream</span>
        <span className={`status-badge status-badge--stream status-badge--${streamStatus}`}>
          {STREAM_LABELS[streamStatus]}
        </span>
      </div>
      <div className="status-row">
        <span className="status-label">Detection</span>
        <span
          className={`status-badge status-badge--detection status-badge--${detectionStatus}`}
        >
          {DETECTION_LABELS[detectionStatus]}
        </span>
      </div>
      <div className="status-row">
        <span className="status-label">Viewers</span>
        <span className="status-value">{activeViewerCount}</span>
      </div>
      {rtcState === 'reconnecting' && (
        <div className="status-message status-message--warn">
          Video reconnecting; stream may be temporarily unavailable.
        </div>
      )}
      {rtcError && (rtcState === 'error' || rtcState === 'reconnecting') && (
        <div className="status-message status-message--error">
          Playback error: {rtcError}
        </div>
      )}
      {rtcState === 'live' && streamStatus === 'stream_unavailable' && (
        <div className="status-message status-message--warn">
          Live video is connected, but backend detection cannot read the RTSP source.
        </div>
      )}
      {detectionStatus === 'stale' && (
        <div className="status-message status-message--info">
          Detection stale; live video remains active.
        </div>
      )}
      {detectionStatus === 'unavailable' && streamStatus === 'live' && (
        <div className="status-message status-message--info">
          Detection unavailable; live video continues.
        </div>
      )}
    </div>
  )
}
