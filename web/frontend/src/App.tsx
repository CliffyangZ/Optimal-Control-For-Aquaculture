import { useCallback, useEffect, useRef, useState } from 'react'
import { MonitoringControls } from './components/MonitoringControls'
import { VideoStage } from './components/VideoStage'
import { StatusPanel } from './components/StatusPanel'
import { useWebRtcPlayer } from './hooks/useWebRtcPlayer'
import { useOverlaySocket } from './hooks/useOverlaySocket'
import { OverlayLayer } from './components/OverlayLayer'
import { DetectionPanel } from './components/DetectionPanel'
import { getConfig, startSession, stopSession, getStatus } from './services/apiClient'
import type { RuntimeConfig, SessionStatus, StreamStatus, DetectionStatus } from './types/streaming'

const POLL_INTERVAL_MS = 5000

export default function App() {
  const [config, setConfig] = useState<RuntimeConfig | null>(null)
  const [session, setSession] = useState<SessionStatus | null>(null)
  const [viewerId, setViewerId] = useState<string | null>(null)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [overlayEnabled, setOverlayEnabled] = useState(true)

  const videoRef = useRef<HTMLVideoElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const streamStatus: StreamStatus = session?.stream_status ?? 'waiting'
  const detectionStatus: DetectionStatus = session?.detection_status ?? 'unavailable'
  const activeViewerCount = session?.active_viewer_count ?? 0

  const { state: rtcState, error: rtcError, stop: stopRtc } = useWebRtcPlayer({
    whepUrl: isMonitoring && config ? config.webrtc_url : null,
    videoRef,
    enabled: isMonitoring,
  })

  const { latestDetections } = useOverlaySocket({
    wsUrl: isMonitoring && config ? config.overlay_ws_url : null,
    viewerId: viewerId ?? '',
    enabled: isMonitoring,
    staleMs: config?.stale_overlay_ms ?? 500,
    onStatus: (msg) => {
      setSession(prev => prev ? {
        ...prev,
        stream_status: msg.stream_status,
        detection_status: msg.detection_status,
        active_viewer_count: msg.active_viewer_count,
        message: msg.message,
      } : prev)
    },
  })

  const effectiveDetectionStatus: DetectionStatus = latestDetections?.detection_status ?? detectionStatus

  // Load config on mount
  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(err => console.error('config load failed', err))
  }, [])

  // Poll status while monitoring
  useEffect(() => {
    if (!isMonitoring) {
      if (pollRef.current) clearInterval(pollRef.current)
      return
    }
    pollRef.current = setInterval(async () => {
      try {
        const s = await getStatus()
        setSession(s)
      } catch { /* ignore poll errors */ }
    }, POLL_INTERVAL_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [isMonitoring])

  const handleStart = useCallback(async () => {
    setIsLoading(true)
    try {
      const s = await startSession()
      setViewerId(parseViewerId(s.message))
      setSession(s)
      setIsMonitoring(true)
    } catch (err) {
      console.error('start session failed', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleStop = useCallback(async () => {
    setIsMonitoring(false)
    stopRtc()
    try {
      if (viewerId) await stopSession(viewerId)
      else await stopSession()
    } catch { /* ignore */ }
    setViewerId(null)
    setSession(null)
  }, [viewerId, stopRtc])

  const handleToggleOverlay = useCallback(() => {
    setOverlayEnabled(v => !v)
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Aquaculture Monitoring</h1>
        <MonitoringControls
          isMonitoring={isMonitoring}
          isLoading={isLoading}
          onStart={handleStart}
          onStop={handleStop}
          overlayEnabled={overlayEnabled}
          onToggleOverlay={handleToggleOverlay}
        />
      </header>

      <main className="app-main">
        <div className="video-area">
          <VideoStage ref={videoRef} rtcState={rtcState} streamStatus={streamStatus}>
            {overlayEnabled && latestDetections && (
              <OverlayLayer
                overlay={latestDetections}
                videoRef={videoRef}
                staleMs={config?.stale_overlay_ms ?? 500}
              />
            )}
          </VideoStage>
        </div>

        <aside className="app-sidebar">
          <StatusPanel
            streamStatus={streamStatus}
            detectionStatus={effectiveDetectionStatus}
            rtcState={rtcState}
            rtcError={rtcError}
            activeViewerCount={activeViewerCount}
          />
          <DetectionPanel
            overlay={latestDetections}
            staleMs={config?.stale_overlay_ms ?? 500}
            detectionStatus={effectiveDetectionStatus}
            statusMessage={session?.message}
          />
        </aside>
      </main>
    </div>
  )
}

function parseViewerId(message: string | null): string | null {
  if (!message?.startsWith('viewer_id:')) return null
  return message.slice('viewer_id:'.length)
}
