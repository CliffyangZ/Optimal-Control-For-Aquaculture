import type { DetectionPayload, DetectionsMessage, DetectionStatus } from '../types/streaming'

interface DetectionPanelProps {
  overlay: DetectionsMessage | null
  staleMs: number
  detectionStatus: DetectionStatus
  statusMessage?: string | null
}

type SummaryGroup = 'fishLike' | 'feed' | 'debris'

export function DetectionPanel({ overlay, staleMs, detectionStatus, statusMessage }: DetectionPanelProps) {
  const isStale = overlay ? Date.now() - overlay.timestamp_ms > staleMs : detectionStatus === 'stale'
  const effectiveStatus = isStale ? 'stale' : detectionStatus
  const detections = effectiveStatus === 'active' ? overlay?.detections ?? [] : []
  const summary = summarizeDetections(detections)

  return (
    <div className="detection-panel">
      <p className="detection-panel__title">YOLO Detection</p>
      <p className={`status-message status-message--${effectiveStatus === 'active' ? 'info' : 'warn'}`}>
        {statusText(effectiveStatus, detections.length, statusMessage)}
      </p>
      <div className="detection-panel__summary" aria-label="detection summary">
        <SummaryItem label="Total" value={summary.total.toString()} />
        <SummaryItem label="Fish-like" value={formatGroup(summary, 'fishLike')} />
        <SummaryItem label="Feed" value={formatGroup(summary, 'feed')} />
        <SummaryItem label="Debris" value={formatGroup(summary, 'debris')} />
      </div>
    </div>
  )
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="detection-panel__summary-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function summarizeDetections(detections: DetectionPayload[]) {
  const groups: Record<SummaryGroup, { count: number; highest: number | null }> = {
    fishLike: { count: 0, highest: null },
    feed: { count: 0, highest: null },
    debris: { count: 0, highest: null },
  }

  for (const detection of detections) {
    const group = detection.class_name === 'feed'
      ? 'feed'
      : detection.class_name === 'debris'
        ? 'debris'
        : 'fishLike'
    groups[group].count += 1
    groups[group].highest = Math.max(groups[group].highest ?? 0, detection.confidence)
  }

  return { total: detections.length, groups }
}

function formatGroup(summary: ReturnType<typeof summarizeDetections>, group: SummaryGroup): string {
  const item = summary.groups[group]
  if (item.count === 0) return '0'
  return `${item.count} / ${Math.round((item.highest ?? 0) * 100)}% max`
}

function statusText(status: DetectionStatus, count: number, statusMessage?: string | null): string {
  if (status === 'loading') return statusMessage ?? 'Loading detection model'
  if (status === 'unavailable') return statusMessage ?? 'Detection model unavailable'
  if (status === 'stale') return 'Detection data is stale'
  if (count === 0) return 'No fish, feed, or debris detected'
  return 'Active YOLO detections'
}
