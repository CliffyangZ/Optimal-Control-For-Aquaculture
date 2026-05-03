import { describe, it, expect } from 'vitest'

describe('Responsive layout', () => {
  it('MonitoringControls renders with overlay toggle when monitoring', async () => {
    const { MonitoringControls } = await import('./MonitoringControls')
    const { render, screen } = await import('@testing-library/react')
    render(
      <MonitoringControls
        isMonitoring={true}
        isLoading={false}
        onStart={() => {}}
        onStop={() => {}}
        overlayEnabled={true}
        onToggleOverlay={() => {}}
      />
    )
    expect(screen.getByRole('button', { name: /stop/i })).toBeDefined()
    expect(screen.getByRole('button', { name: /overlay/i })).toBeDefined()
  })

  it('MonitoringControls shows Start when not monitoring', async () => {
    const { MonitoringControls } = await import('./MonitoringControls')
    const { render, screen } = await import('@testing-library/react')
    render(
      <MonitoringControls
        isMonitoring={false}
        isLoading={false}
        onStart={() => {}}
        onStop={() => {}}
        overlayEnabled={false}
        onToggleOverlay={() => {}}
      />
    )
    expect(screen.getByRole('button', { name: /start/i })).toBeDefined()
  })
})

describe('Unsupported browser messaging', () => {
  it('browserSupport returns unsupported when RTCPeerConnection missing', async () => {
    const savedRTC = (globalThis as Record<string, unknown>).RTCPeerConnection
    delete (globalThis as Record<string, unknown>).RTCPeerConnection

    const { checkBrowserSupport } = await import('../services/browserSupport')
    const result = checkBrowserSupport()
    expect(result.supported).toBe(false)
    expect(result.reason).toContain('WebRTC')

    ;(globalThis as Record<string, unknown>).RTCPeerConnection = savedRTC
  })
})
