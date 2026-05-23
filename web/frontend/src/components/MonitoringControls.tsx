interface MonitoringControlsProps {
  isMonitoring: boolean
  isLoading: boolean
  onStart: () => void
  onStop: () => void
  overlayEnabled: boolean
  onToggleOverlay: () => void
}

export function MonitoringControls({
  isMonitoring,
  isLoading,
  onStart,
  onStop,
  overlayEnabled,
  onToggleOverlay,
}: MonitoringControlsProps) {
  return (
    <div className="monitoring-controls">
      {!isMonitoring ? (
        <button
          className="btn btn-start"
          onClick={onStart}
          disabled={isLoading}
        >
          {isLoading ? 'Starting…' : 'Start'}
        </button>
      ) : (
        <button className="btn btn-stop" onClick={onStop}>
          Stop
        </button>
      )}
      {isMonitoring && (
        <button
          className="btn btn-overlay-toggle"
          onClick={onToggleOverlay}
          aria-label={overlayEnabled ? 'Hide detection overlay boxes' : 'Show detection overlay boxes'}
          title={overlayEnabled ? 'Hide detection boxes' : 'Show detection boxes'}
        >
          {overlayEnabled ? 'Hide Boxes' : 'Show Boxes'}
        </button>
      )}
    </div>
  )
}
