import { useEffect, useState } from 'react';
import { getConfig, getStatus, heartbeat, startSession, stopSession } from './services/apiClient';
import { useOverlaySocket } from './hooks/useOverlaySocket';
import { useWebRtcPlayer } from './hooks/useWebRtcPlayer';
import { VideoStage } from './components/VideoStage';
import { StatusPanel } from './components/StatusPanel';
import type { MonitorConfig, SessionStatus } from './types/monitoring';

export function App() {
  const [config, setConfig] = useState<MonitorConfig | null>(null);
  const [session, setSession] = useState<SessionStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const viewerId = session?.viewer_id ?? null;
  const { videoRef, ready: videoReady, error: playbackError } = useWebRtcPlayer(config?.webrtc_url ?? null);
  const { message: overlay, connected: socketConnected, error: socketError } = useOverlaySocket(config?.overlay_ws_url ?? null, viewerId);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        const [loadedConfig, initialStatus] = await Promise.all([getConfig(), getStatus()]);
        if (cancelled) return;
        setConfig(loadedConfig);
        setSession(initialStatus);
        const started = await startSession();
        if (!cancelled) setSession(started);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to start monitor');
      }
    }
    void boot();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!viewerId) return;
    const interval = window.setInterval(() => { void heartbeat(viewerId); }, 10_000);
    const stop = () => { void stopSession(viewerId); };
    window.addEventListener('beforeunload', stop);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener('beforeunload', stop);
      void stopSession(viewerId);
    };
  }, [viewerId]);

  return (
    <main className="page-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Aquaculture CV</p>
          <h1>Live Fish Detection Monitor</h1>
        </div>
        <div className="source-pill">{config?.source_id ?? 'loading source'}</div>
      </section>
      {error && <div className="warning banner">{error}</div>}
      <section className="monitor-grid">
        <VideoStage videoRef={videoRef} detections={overlay?.detections ?? []} />
        <StatusPanel
          config={config}
          session={session}
          overlay={overlay}
          videoReady={videoReady}
          playbackError={playbackError}
          socketConnected={socketConnected}
          socketError={socketError}
        />
      </section>
    </main>
  );
}
