import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { OverlayLayer } from './OverlayLayer'
import type { DetectionsMessage } from '../types/streaming'

function makeDetections(overrides?: Partial<DetectionsMessage>): DetectionsMessage {
  return {
    schema: 'detections.v1',
    type: 'detections',
    source_id: 'drone_live',
    frame_id: 1,
    timestamp_ms: Date.now(),
    source_width: 1280,
    source_height: 720,
    stream_status: 'live',
    detection_status: 'active',
    detections: [{ class_name: 'fish', confidence: 0.84, x: 0.1, y: 0.1, width: 0.2, height: 0.2 }],
    message: null,
    ...overrides,
  }
}

function videoRef() {
  const video = document.createElement('video')
  Object.defineProperty(video, 'clientWidth', { value: 1280 })
  Object.defineProperty(video, 'clientHeight', { value: 720 })
  Object.defineProperty(video, 'videoWidth', { value: 1280 })
  Object.defineProperty(video, 'videoHeight', { value: 720 })
  return { current: video } as React.RefObject<HTMLVideoElement | null>
}

describe('OverlayLayer', () => {
  it('does not render SVG when detections are stale', () => {
    const { container } = render(
      <OverlayLayer overlay={makeDetections({ timestamp_ms: Date.now() - 1000 })} videoRef={videoRef()} staleMs={500} />,
    )

    expect(container.querySelector('svg')).toBeNull()
  })

  it('renders raw label and confidence percentage', async () => {
    render(<OverlayLayer overlay={makeDetections()} videoRef={videoRef()} staleMs={500} />)

    expect(await screen.findByText('fish 84%')).toBeInTheDocument()
  })

  it('renders nothing when there are no detections', () => {
    const { container } = render(
      <OverlayLayer overlay={makeDetections({ detections: [] })} videoRef={videoRef()} staleMs={500} />,
    )

    expect(container.querySelector('svg')).toBeNull()
  })
})
