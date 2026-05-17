import { useCallback, useEffect, useRef, useState } from 'react'

export type WebRtcState =
  | 'idle'
  | 'connecting'
  | 'live'
  | 'reconnecting'
  | 'stopped'
  | 'unsupported'
  | 'error'

export const isSupported = typeof RTCPeerConnection !== 'undefined'

interface UseWebRtcPlayerOptions {
  whepUrl: string | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  enabled: boolean
}

interface UseWebRtcPlayerResult {
  state: WebRtcState
  error: string | null
  stop: () => void
  restart: () => void
}

const RECONNECT_DELAY_MS = 3000
const MAX_RECONNECT_DELAY_MS = 30000
// Interval between currentTime checks; must stall for this duration to trigger reconnect.
const VIDEO_STALL_CHECK_MS = 4000

export function useWebRtcPlayer({
  whepUrl,
  videoRef,
  enabled,
}: UseWebRtcPlayerOptions): UseWebRtcPlayerResult {
  const [state, setState] = useState<WebRtcState>(() =>
    isSupported ? 'idle' : 'unsupported'
  )
  const [error, setError] = useState<string | null>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)
  const stoppedRef = useRef(false)
  const reconnectDelayRef = useRef(RECONNECT_DELAY_MS)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const stallTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const cleanup = useCallback(() => {
    if (stallTimerRef.current) {
      clearInterval(stallTimerRef.current)
      stallTimerRef.current = null
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
    if (pcRef.current) {
      pcRef.current.close()
      pcRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [videoRef])

  const connect = useCallback(async () => {
    if (!isSupported || !whepUrl || stoppedRef.current) return

    cleanup()
    setState('connecting')
    setError(null)

    try {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      })
      pcRef.current = pc

      pc.addTransceiver('video', { direction: 'recvonly' })
      pc.addTransceiver('audio', { direction: 'recvonly' })

      pc.ontrack = (event) => {
        if (videoRef.current && event.streams[0]) {
          videoRef.current.srcObject = event.streams[0]
          videoRef.current.play().catch(() => {/* autoplay policy */})
          setState('live')
          reconnectDelayRef.current = RECONNECT_DELAY_MS

          // Monitor video.currentTime to detect a frozen/stalled stream.
          // WebRTC connections can stay "connected" at the ICE level while the
          // video track stops delivering frames (jitter-buffer underrun, RTSP
          // source reconnect, etc.). Checking currentTime every VIDEO_STALL_CHECK_MS
          // catches this without relying on onconnectionstatechange alone.
          if (stallTimerRef.current) clearInterval(stallTimerRef.current)
          let lastCurrentTime = -1
          stallTimerRef.current = setInterval(() => {
            const video = videoRef.current
            if (!video || stoppedRef.current) return
            const cur = video.currentTime
            // Skip check until playback has actually started
            if (lastCurrentTime < 0) {
              if (cur > 0) lastCurrentTime = cur
              return
            }
            if (cur === lastCurrentTime && !video.paused) {
              if (stallTimerRef.current) {
                clearInterval(stallTimerRef.current)
                stallTimerRef.current = null
              }
              if (!stoppedRef.current) {
                setState('reconnecting')
                scheduleReconnect()
              }
            }
            lastCurrentTime = cur
          }, VIDEO_STALL_CHECK_MS)
        }
      }

      pc.onconnectionstatechange = () => {
        const s = pc.connectionState
        if (s === 'failed' || s === 'disconnected') {
          if (!stoppedRef.current) {
            setState('reconnecting')
            scheduleReconnect()
          }
        }
      }

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      const response = await fetch(whepUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp' },
        body: offer.sdp,
      })

      if (!response.ok) {
        throw new Error(`WHEP ${response.status}: ${response.statusText}`)
      }

      const answerSdp = await response.text()
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
    } catch (err) {
      if (!stoppedRef.current) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        setState('reconnecting')
        scheduleReconnect()
      }
    }
  }, [whepUrl, videoRef, cleanup]) // eslint-disable-line react-hooks/exhaustive-deps

  function scheduleReconnect() {
    const delay = reconnectDelayRef.current
    reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS)
    reconnectTimerRef.current = setTimeout(() => {
      if (!stoppedRef.current) connect()
    }, delay)
  }

  const stop = useCallback(() => {
    stoppedRef.current = true
    cleanup()
    setState('stopped')
  }, [cleanup])

  const restart = useCallback(() => {
    stoppedRef.current = false
    reconnectDelayRef.current = RECONNECT_DELAY_MS
    connect()
  }, [connect])

  useEffect(() => {
    if (!enabled || !whepUrl) {
      cleanup()
      setState(isSupported ? 'idle' : 'unsupported')
      return
    }
    stoppedRef.current = false
    connect()
    return () => {
      stoppedRef.current = true
      cleanup()
    }
  }, [enabled, whepUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  return { state, error, stop, restart }
}
