import { useEffect, useState } from 'react';
import type { DetectionsMessage } from '../types/monitoring';

export function useOverlaySocket(url: string | null, viewerId: string | null) {
  const [message, setMessage] = useState<DetectionsMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!url) return;
    const wsUrl = viewerId ? `${url}?viewer_id=${encodeURIComponent(viewerId)}` : url;
    const socket = new WebSocket(wsUrl);
    setError(null);
    setConnected(false);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setError('Overlay connection failed');
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'detections') setMessage(data);
    };

    return () => socket.close();
  }, [url, viewerId]);

  return { message, connected, error };
}
