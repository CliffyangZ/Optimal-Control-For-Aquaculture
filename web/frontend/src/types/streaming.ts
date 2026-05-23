export type StreamStatus =
  | 'waiting'
  | 'connecting'
  | 'live'
  | 'reconnecting'
  | 'stream_unavailable'
  | 'degraded'
  | 'error'

export type DetectionStatus = 'loading' | 'active' | 'stale' | 'unavailable'

export type DetectionClassName = 'fish' | 'lumpfish' | 'feed' | 'debris'

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

export interface DetectionPayload {
  class_name: DetectionClassName
  confidence: number
  x: number
  y: number
  width: number
  height: number
}

export interface DetectionsMessage {
  schema: 'detections.v1'
  type: 'detections'
  source_id: string
  frame_id: number
  timestamp_ms: number
  source_width: number
  source_height: number
  stream_status: StreamStatus
  detection_status: DetectionStatus
  detections: DetectionPayload[]
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

export type WsServerMessage = DetectionsMessage | StatusWsMessage
