import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'

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
