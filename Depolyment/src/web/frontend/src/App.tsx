import { useState } from 'react'
import { VideoPlayer } from './components/VideoPlayer'
import { OverlayCanvas } from './components/OverlayCanvas'
import { HungerBadge } from './components/HungerBadge'
import { FeedAreaChart } from './components/FeedAreaChart'
import { StatusPanel } from './components/StatusPanel'
import { useConfig } from './hooks/useConfig'
import { useSession } from './hooks/useSession'
import { useOverlayWs } from './hooks/useOverlayWs'
import { AnalyticsMessage, DetectionsMessage, StatusMessage } from './types'
import styles from './App.module.css'

function App() {
  const { config, error: configError, loading } = useConfig()
  const { viewerId } = useSession()
  const [detections, setDetections] = useState<DetectionsMessage | null>(null)
  const [analytics, setAnalytics] = useState<AnalyticsMessage | null>(null)
  const [status, setStatus] = useState<StatusMessage | null>(null)

  useOverlayWs(
    config?.overlay_ws_url || 'ws://localhost/ws/overlays',
    {
      onDetections: setDetections,
      onAnalytics: setAnalytics,
      onStatus: setStatus,
      onError: (err) => console.error('WebSocket error:', err),
    }
  )

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading configuration...</div>
      </div>
    )
  }

  if (configError || !config) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          Failed to load config: {configError || 'Unknown error'}
        </div>
      </div>
    )
  }

  const videoWidth = detections?.source_width || 1280
  const videoHeight = detections?.source_height || 720

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>🐟 RF-DETR Aquaculture Monitor</h1>
        <p>Real-time fish monitoring with WebRTC</p>
      </header>

      <div className={styles.grid}>
        <div className={styles.left}>
          <div className={styles.videoStage}>
            <VideoPlayer whepUrl={config.webrtc_url} />
            <OverlayCanvas
              width={videoWidth}
              height={videoHeight}
              detections={detections}
            />
          </div>
          <StatusPanel detections={detections} status={status} />
        </div>

        <div className={styles.right}>
          <HungerBadge analytics={analytics} />
          <FeedAreaChart analytics={analytics} />
        </div>
      </div>

      {!viewerId && (
        <div className={styles.warning}>Connecting to server...</div>
      )}
    </div>
  )
}

export default App
