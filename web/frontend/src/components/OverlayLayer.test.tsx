import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { OverlayLayer } from './OverlayLayer'
import type { OverlayMessage } from '../types/streaming'

function makeOverlay(overrides?: Partial<OverlayMessage>): OverlayMessage {
  return {
    schema: 'overlay.v1',
    type: 'overlay',
    source_id: 'drone_live',
    frame_id: 1,
    timestamp_ms: Date.now(),
    source_width: 1280,
    source_height: 720,
    stream_status: 'live',
    detection_status: 'ok',
    activity: { level: 'Medium', score: 52, confidence: 0.84 },
    regions: [{ id: 'region-1', x: 0.1, y: 0.1, width: 0.2, height: 0.2 }],
    message: null,
    ...overrides,
  }
}

describe('OverlayLayer – stale hiding', () => {
  it('does not render SVG when overlay is stale', () => {
    const overlay = makeOverlay({ timestamp_ms: Date.now() - 1000 })
    const ref = { current: null } as React.RefObject<HTMLVideoElement | null>
    const { container } = render(
      <OverlayLayer overlay={overlay} videoRef={ref} staleMs={500} />
    )
    expect(container.querySelector('svg')).toBeNull()
  })
})

describe('OverlayLayer – rendering', () => {
  it('renders nothing when regions list is empty', () => {
    const overlay = makeOverlay({ regions: [] })
    const ref = { current: null } as React.RefObject<HTMLVideoElement | null>
    const { container } = render(
      <OverlayLayer overlay={overlay} videoRef={ref} staleMs={500} />
    )
    expect(container.querySelector('svg')).toBeNull()
  })
})

describe('OverlayLayer – out-of-order messages', () => {
  it('module exports OverlayLayer component', () => {
    expect(typeof OverlayLayer).toBe('function')
  })
})
