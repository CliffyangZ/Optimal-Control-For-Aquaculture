import type { DetectionPayload, DetectionsMessage, MonitorConfig, SessionStatus } from '../types/monitoring';

function countByClass(detections: DetectionPayload[]) {
  return detections.reduce<Record<string, number>>((counts, detection) => {
    counts[detection.class_name] = (counts[detection.class_name] ?? 0) + 1;
    return counts;
  }, {});
}

export function StatusPanel({ config, session, overlay, videoReady, playbackError, socketConnected, socketError }: {
  config: MonitorConfig | null;
  session: SessionStatus | null;
  overlay: DetectionsMessage | null;
  videoReady: boolean;
  playbackError: string | null;
  socketConnected: boolean;
  socketError: string | null;
}) {
  const counts = countByClass(overlay?.detections ?? []);
  return (
    <aside className="status-panel">
      <h2>Live RF-DETR</h2>
      <dl>
        <dt>Video</dt><dd>{videoReady ? 'live' : playbackError ? 'error' : 'connecting'}</dd>
        <dt>Stream</dt><dd>{overlay?.stream_status ?? session?.stream_status ?? 'waiting'}</dd>
        <dt>Detection</dt><dd>{overlay?.detection_status ?? session?.detection_status ?? 'unavailable'}</dd>
        <dt>Overlay socket</dt><dd>{socketConnected ? 'connected' : socketError ?? 'connecting'}</dd>
        <dt>FPS</dt><dd>{(overlay?.detection_fps ?? session?.detection_fps ?? 0).toFixed(1)}</dd>
        <dt>Inference</dt><dd>{overlay ? `${overlay.inference_ms.toFixed(0)} ms` : '-'}</dd>
        <dt>Viewers</dt><dd>{session?.active_viewer_count ?? 0}</dd>
        <dt>Device</dt><dd>{config?.device ?? '-'}</dd>
      </dl>
      {(playbackError || socketError || session?.message) && <p className="warning">{playbackError ?? socketError ?? session?.message}</p>}
      <h3>Class Counts</h3>
      {Object.keys(counts).length === 0 ? <p className="muted">No detections</p> : (
        <ul className="class-counts">
          {Object.entries(counts).map(([name, count]) => <li key={name}><span>{name}</span><strong>{count}</strong></li>)}
        </ul>
      )}
    </aside>
  );
}
