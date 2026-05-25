export type StreamStatus = 'waiting' | 'connecting' | 'live' | 'reconnecting' | 'stream_unavailable' | 'error' | 'degraded';
export type DetectionStatus = 'loading' | 'active' | 'stale' | 'unavailable';

export interface MonitorConfig {
  source_id: string;
  webrtc_url: string;
  overlay_ws_url: string;
  stale_overlay_ms: number;
  local_trusted_network_only: boolean;
  detector_model_configured: boolean;
  confidence_threshold: number;
  device: string;
}

export interface SessionStatus {
  source_id: string;
  stream_status: StreamStatus;
  detection_status: DetectionStatus;
  active_viewer_count: number;
  detection_fps: number;
  last_frame_at: string | null;
  last_overlay_at: string | null;
  message: string;
  viewer_id: string | null;
}

export interface DetectionPayload {
  class_name: string;
  class_index: number | null;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionsMessage {
  schema: 'detections.v1';
  type: 'detections';
  source_id: string;
  frame_id: number;
  timestamp_ms: number;
  source_width: number;
  source_height: number;
  stream_status: StreamStatus;
  detection_status: DetectionStatus;
  detections: DetectionPayload[];
  inference_ms: number;
  detection_fps: number;
  message: string;
}
