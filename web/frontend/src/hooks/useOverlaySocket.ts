import { useCallback, useEffect, useRef, useState } from 'react'
import type { OverlayMessage, StatusWsMessage } from '../types/streaming'

interface UseOverlaySocketOptions {
  wsUrl: string | null
  viewerId: string
  enabled: boolean
  staleMs: number
  onStatus?: (msg: StatusWsMessage) => void
}

interface UseOverlaySocketResult {
  latestOverlay: OverlayMessage | null
  connected: boolean
}

const HEARTBEAT_INTERVAL_MS = 10000
const BASE_RECONNECT_MS = 2000
const MAX_RECONNECT_MS = 30000

export function useOverlaySocket({
  wsUrl,
  viewerId,
  enabled,
  staleMs,
  onStatus,
}: UseOverlaySocketOptions): UseOverlaySocketResult {
  const [latestOverlay, setLatestOverlay] = useState<OverlayMessage | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayRef = useRef(BASE_RECONNECT_MS)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const stoppedRef = useRef(false)
  const latestTimestampRef = useRef(0)

  const clearHeartbeat = () => {
    if (heartbeatTimerRef.current) clearInterval(heartbeatTimerRef.current)
    heartbeatTimerRef.current = null
  }

  const clearReconnect = () => {
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
    reconnectTimerRef.current = null
  }

  const disconnect = useCallback(() => {
    clearHeartbeat()
    clearReconnect()
    if (wsRef.current) {
      const ws = wsRef.current
      wsRef.current = null
      ws.close()
    }
    setConnected(false)
  }, [])

  const connect = useCallback(() => {
    if (!wsUrl || !viewerId || stoppedRef.current) return
    disconnect()

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'viewer.hello', viewer_id: viewerId, overlay_enabled: true }))
      reconnectDelayRef.current = BASE_RECONNECT_MS
      setConnected(true)
      heartbeatTimerRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'viewer.heartbeat',
            viewer_id: viewerId,
            timestamp_ms: Date.now(),
          }))
        }
      }, HEARTBEAT_INTERVAL_MS)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string)
        if (data.type === 'overlay') {
          const msg = data as OverlayMessage
          if (msg.timestamp_ms > latestTimestampRef.current) {
            latestTimestampRef.current = msg.timestamp_ms
            setLatestOverlay(msg)
          }
        } else if (data.type === 'status') {
          onStatus?.(data as StatusWsMessage)
        }
      } catch { /* ignore malformed */ }
    }

    ws.onclose = () => {
      clearHeartbeat()
      setConnected(false)
      if (!stoppedRef.current) {
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_MS)
        reconnectTimerRef.current = setTimeout(() => {
          if (!stoppedRef.current) connect()
        }, delay)
      }
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [wsUrl, viewerId, disconnect, onStatus])

  // Stale overlay cleanup
  useEffect(() => {
    if (!latestOverlay) return
    const timer = setTimeout(() => {
      setLatestOverlay(prev => {
        if (!prev) return null
        const age = Date.now() - prev.timestamp_ms
        return age > staleMs ? null : prev
      })
    }, staleMs)
    return () => clearTimeout(timer)
  }, [latestOverlay, staleMs])

  useEffect(() => {
    if (!enabled || !wsUrl) {
      stoppedRef.current = true
      disconnect()
      setLatestOverlay(null)
      latestTimestampRef.current = 0
      return
    }
    stoppedRef.current = false
    connect()
    return () => {
      stoppedRef.current = true
      disconnect()
    }
  }, [enabled, wsUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  return { latestOverlay, connected }
}
