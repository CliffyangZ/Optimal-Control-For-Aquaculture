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

  const cleanup = useCallback(() => {
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
