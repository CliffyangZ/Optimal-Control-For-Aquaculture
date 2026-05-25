import { useEffect, useRef, useState } from 'react';

export function useWebRtcPlayer(webrtcUrl: string | null) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!webrtcUrl || !videoRef.current) return;
    let cancelled = false;
    const pc = new RTCPeerConnection();
    peerRef.current = pc;
    setError(null);
    setReady(false);

    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
        setReady(true);
      }
    };

    async function connect() {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        const response = await fetch(webrtcUrl!, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp ?? '',
        });
        if (!response.ok) throw new Error(`WHEP failed: ${response.status}`);
        const answer = await response.text();
        if (!cancelled) await pc.setRemoteDescription({ type: 'answer', sdp: answer });
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'WebRTC playback failed');
      }
    }

    void connect();
    return () => {
      cancelled = true;
      pc.close();
      peerRef.current = null;
      setReady(false);
    };
  }, [webrtcUrl]);

  return { videoRef, ready, error };
}
