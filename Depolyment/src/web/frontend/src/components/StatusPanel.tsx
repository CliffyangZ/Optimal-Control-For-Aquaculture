import { DetectionsMessage, StatusMessage } from '../types'
import styles from './StatusPanel.module.css'

interface StatusPanelProps {
  detections: DetectionsMessage | null
  status: StatusMessage | null
}

export function StatusPanel({ detections, status }: StatusPanelProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.row}>
        <span>Inference FPS</span>
        <span>{detections ? '~1' : '—'}</span>
      </div>
      <div className={styles.row}>
        <span>Detections</span>
        <span>{detections?.detections.length || 0}</span>
      </div>
      <div className={styles.row}>
        <span>Stream Status</span>
        <span className={styles.status}>{status?.stream_status || '—'}</span>
      </div>
      <div className={styles.row}>
        <span>Detection Status</span>
        <span className={styles.status}>{status?.detection_status || '—'}</span>
      </div>
      <div className={styles.row}>
        <span>Viewers</span>
        <span>{status?.active_viewer_count || 0}</span>
      </div>
    </div>
  )
}
