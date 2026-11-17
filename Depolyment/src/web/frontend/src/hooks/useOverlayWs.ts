import { useEffect, useCallback } from 'react'
import { AnalyticsMessage, DetectionsMessage, StatusMessage, WebSocketMessage } from '../types'

interface Handlers {
  onDetections?: (msg: DetectionsMessage) => void
  onAnalytics?: (msg: AnalyticsMessage) => void
  onStatus?: (msg: StatusMessage) => void
  onError?: (error: Error) => void
}

export function useOverlayWs(url: string, handlers: Handlers) {
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const msg: WebSocketMessage = JSON.parse(event.data)

      switch (msg.schema) {
        case 'detections.v1':
          handlers.onDetections?.(msg as DetectionsMessage)
          break
        case 'analytics.v1':
          handlers.onAnalytics?.(msg as AnalyticsMessage)
          break
        case 'status.v1':
          handlers.onStatus?.(msg as StatusMessage)
          break
      }
    } catch (err) {
      handlers.onError?.(err instanceof Error ? err : new Error('Unknown error'))
    }
  }, [handlers])

  useEffect(() => {
    const ws = new WebSocket(url)

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = handleMessage

    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      handlers.onError?.(new Error('WebSocket error'))
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
    }

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
    }
  }, [url, handleMessage, handlers])
}
