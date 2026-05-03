export interface BrowserSupportResult {
  supported: boolean
  reason: string | null
}

export function checkBrowserSupport(): BrowserSupportResult {
  if (typeof RTCPeerConnection === 'undefined') {
    return {
      supported: false,
      reason:
        'WebRTC is not available in this browser. Please use Chrome or Edge for live stream playback.',
    }
  }
  if (typeof fetch === 'undefined') {
    return { supported: false, reason: 'Fetch API is not available in this browser.' }
  }
  if (typeof WebSocket === 'undefined') {
    return { supported: false, reason: 'WebSocket is not available in this browser.' }
  }
  return { supported: true, reason: null }
}
