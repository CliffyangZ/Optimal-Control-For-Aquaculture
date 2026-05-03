export type StreamStatus =
  | 'waiting'
  | 'connecting'
  | 'live'
  | 'reconnecting'
  | 'stream_unavailable'
  | 'degraded'

export type DetectionStatus = 'ok' | 'delayed' | 'degraded' | 'unavailable'

export type ActivityLevel = 'Low' | 'Medium' | 'High'

export interface RuntimeConfig {
  source_id: string
  webrtc_url: string
  overlay_ws_url: string
  stale_overlay_ms: number
  local_trusted_network_only: boolean
  primary_browser_support: string
}

export interface SessionStatus {
  source_id: string
  stream_status: StreamStatus
  detection_status: DetectionStatus
  active_viewer_count: number
  started_at: string | null
  last_frame_at: string | null
  last_overlay_at: string | null
  detection_fps: number
  message: string | null
}

export interface ActivityBadge {
  level: ActivityLevel
  score: number
  confidence: number
}

export interface OverlayRegion {
  id: string
  x: number
  y: number
  width: number
  height: number
}

export interface OverlayMessage {
  schema: 'overlay.v1'
  type: 'overlay'
  source_id: string
  frame_id: number
  timestamp_ms: number
  source_width: number
  source_height: number
  stream_status: StreamStatus
  detection_status: DetectionStatus
  activity: ActivityBadge | null
  regions: OverlayRegion[]
  message: string | null
}

export interface StatusWsMessage {
  schema: 'status.v1'
  type: 'status'
  source_id: string
  timestamp_ms: number
  stream_status: StreamStatus
  detection_status: DetectionStatus
  active_viewer_count: number
  message: string | null
}

export type WsServerMessage = OverlayMessage | StatusWsMessage
