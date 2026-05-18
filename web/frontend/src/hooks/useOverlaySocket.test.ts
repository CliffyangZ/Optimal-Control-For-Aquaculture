import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useOverlaySocket } from './useOverlaySocket'

class FakeWebSocket {
  static instance: FakeWebSocket | null = null
  static OPEN = 1
  readyState = FakeWebSocket.OPEN
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  sent: string[] = []

  constructor() {
    FakeWebSocket.instance = this
  }

  send(message: string) {
    this.sent.push(message)
  }

  close() {
    this.onclose?.()
  }
}

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
  FakeWebSocket.instance = null
})

describe('useOverlaySocket', () => {
  it('keeps latest detections and clears stale detections', () => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeWebSocket)

    const { result } = renderHook(() => useOverlaySocket({
      wsUrl: 'ws://localhost:8000/ws/overlays',
      viewerId: 'viewer-1',
      enabled: true,
      staleMs: 50,
    }))

    act(() => {
      FakeWebSocket.instance?.onopen?.()
      FakeWebSocket.instance?.onmessage?.({
        data: JSON.stringify({
          schema: 'detections.v1',
          type: 'detections',
          source_id: 'drone_live',
          frame_id: 2,
          timestamp_ms: Date.now(),
          source_width: 1280,
          source_height: 720,
          stream_status: 'live',
          detection_status: 'active',
          detections: [],
          message: null,
        }),
      })
      FakeWebSocket.instance?.onmessage?.({
        data: JSON.stringify({
          schema: 'detections.v1',
          type: 'detections',
          source_id: 'drone_live',
          frame_id: 1,
          timestamp_ms: Date.now() - 100,
          source_width: 1280,
          source_height: 720,
          stream_status: 'live',
          detection_status: 'active',
          detections: [{ class_name: 'fish', confidence: 0.8, x: 0, y: 0, width: 0.1, height: 0.1 }],
          message: null,
        }),
      })
    })

    expect(result.current.latestDetections?.frame_id).toBe(2)

    act(() => {
      vi.advanceTimersByTime(60)
    })

    expect(result.current.latestDetections).toBeNull()
  })
})
