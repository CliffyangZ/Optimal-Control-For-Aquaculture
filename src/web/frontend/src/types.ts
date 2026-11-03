export interface DetectionPayload {
  class_name: string;
  class_index?: number;
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
  stream_status: string;
  detection_status: string;
  detections: DetectionPayload[];
}

export interface AnalyticsMessage {
  schema: 'analytics.v1';
  type: 'analytics';
  source_id: string;
  timestamp_ms: number;
  hunger_level: string;
  consumption_rate: number;
  frame_ids: number[];
  raw_area: number[];
  kalman_area: number[];
  ema_area: number[];
}

export interface StatusMessage {
  schema: 'status.v1';
  type: 'status';
  source_id: string;
  timestamp_ms: number;
  stream_status: string;
  detection_status: string;
  active_viewer_count: number;
  message: string;
}

export interface ConfigResponse {
  source_id: string;
  webrtc_url: string;
  overlay_ws_url: string;
  stale_overlay_ms: number;
  local_trusted_network_only: boolean;
}

export interface SessionStatusResponse {
  source_id: string;
  stream_status: string;
  detection_status: string;
  active_viewer_count: number;
}

export type WebSocketMessage = DetectionsMessage | AnalyticsMessage | StatusMessage;
