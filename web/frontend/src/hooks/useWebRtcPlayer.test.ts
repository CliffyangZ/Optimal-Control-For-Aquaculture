import { describe, it, expect } from 'vitest'

describe('useWebRtcPlayer – unsupported browser', () => {
  it('detects unsupported browser when RTCPeerConnection is absent', async () => {
    const originalRTC = (globalThis as Record<string, unknown>).RTCPeerConnection
    delete (globalThis as Record<string, unknown>).RTCPeerConnection

    await import('./useWebRtcPlayer')
    const isSupported = typeof RTCPeerConnection !== 'undefined'

    expect(isSupported).toBe(false)
    ;(globalThis as Record<string, unknown>).RTCPeerConnection = originalRTC
  })
})

describe('useWebRtcPlayer – reconnect state', () => {
  it('hook module exports useWebRtcPlayer', async () => {
    const mod = await import('./useWebRtcPlayer')
    expect(typeof mod.useWebRtcPlayer).toBe('function')
  })
})
