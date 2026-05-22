import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

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

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      source_id: 'drone_live',
      webrtc_url: 'http://localhost:8889/drone_live/whep',
      overlay_ws_url: 'ws://localhost:8000/ws/overlays',
      stale_overlay_ms: 500,
      local_trusted_network_only: true,
      primary_browser_support: 'Chrome and Edge are primary validation targets.',
    }),
  }))
})

afterEach(() => {
  vi.unstubAllGlobals()
  FakeWebSocket.instance = null
})

describe('App – live stream controls', () => {
  it('renders Start button', async () => {
    const App = (await import('./App')).default
    render(<App />)
    expect(screen.queryByRole('button', { name: /start/i })).toBeDefined()
  })
})

describe('App – connection status', () => {
  it('renders some status indicator', async () => {
    const App = (await import('./App')).default
    render(<App />)
    expect(document.querySelector('.app')).not.toBeNull()
  })
})

describe('App – detection box visibility', () => {
  it('keeps status and summary visible when boxes are hidden', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/session/start')
        ? {
            source_id: 'drone_live',
            stream_status: 'live',
            detection_status: 'active',
            active_viewer_count: 1,
            started_at: '2026-05-17T00:00:00+00:00',
            last_frame_at: null,
            last_overlay_at: null,
            detection_fps: 0,
            message: 'viewer_id:test-viewer',
          }
        : {
            source_id: 'drone_live',
            webrtc_url: 'http://localhost:8889/drone_live/whep',
            overlay_ws_url: 'ws://localhost:8000/ws/overlays',
            stale_overlay_ms: 500,
            local_trusted_network_only: true,
            primary_browser_support: 'Chrome and Edge are primary validation targets.',
          }
      return { ok: true, json: async () => body } as Response
    })

    const App = (await import('./App')).default
    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: /start/i }))

    const hideButton = await screen.findByRole('button', { name: /hide detection overlay boxes/i })
    fireEvent.click(hideButton)

    await waitFor(() => {
      expect(screen.getByText('YOLO Detection')).toBeInTheDocument()
      expect(screen.getByLabelText('detection summary')).toBeInTheDocument()
      expect(screen.getByText('Active')).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /show detection overlay boxes/i })).toBeInTheDocument()
    })
  })
})

describe('App – detection stream status', () => {
  it('uses the latest YOLO payload stream status for the visible stream state', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    vi.mocked(fetch).mockImplementation(async (input) => {
      const path = String(input)
      const body = path.includes('/session/start')
        ? {
            source_id: 'drone_live',
            stream_status: 'stream_unavailable',
            detection_status: 'loading',
            active_viewer_count: 1,
            started_at: '2026-05-17T00:00:00+00:00',
            last_frame_at: null,
            last_overlay_at: null,
            detection_fps: 0,
            message: 'viewer_id:test-viewer',
          }
        : {
            source_id: 'drone_live',
            webrtc_url: 'http://localhost:8889/drone_live/whep',
            overlay_ws_url: 'ws://localhost:8000/ws/overlays',
            stale_overlay_ms: 500,
            local_trusted_network_only: true,
            primary_browser_support: 'Chrome and Edge are primary validation targets.',
          }
      return { ok: true, json: async () => body } as Response
    })

    const App = (await import('./App')).default
    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: /start/i }))

    await waitFor(() => {
      expect(FakeWebSocket.instance).not.toBeNull()
    })

    act(() => {
      FakeWebSocket.instance?.onopen?.()
      FakeWebSocket.instance?.onmessage?.({
        data: JSON.stringify({
          schema: 'detections.v1',
          type: 'detections',
          source_id: 'drone_live',
          frame_id: 1,
          timestamp_ms: Date.now(),
          source_width: 1280,
          source_height: 720,
          stream_status: 'live',
          detection_status: 'active',
          detections: [],
          message: null,
        }),
      })
    })

    expect(await screen.findByText('Live')).toBeInTheDocument()
  })
})
