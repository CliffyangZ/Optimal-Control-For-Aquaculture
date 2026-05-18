import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { DetectionPanel } from './DetectionPanel'
import type { DetectionsMessage } from '../types/streaming'

const overlay: DetectionsMessage = {
  schema: 'detections.v1',
  type: 'detections',
  source_id: 'drone_live',
  frame_id: 1,
  timestamp_ms: Date.now(),
  source_width: 1280,
  source_height: 720,
  stream_status: 'live',
  detection_status: 'active',
  detections: [
    { class_name: 'fish', confidence: 0.8, x: 0.1, y: 0.1, width: 0.2, height: 0.2 },
    { class_name: 'lumpfish', confidence: 0.9, x: 0.2, y: 0.2, width: 0.2, height: 0.2 },
    { class_name: 'feed', confidence: 0.7, x: 0.3, y: 0.3, width: 0.2, height: 0.2 },
    { class_name: 'debris', confidence: 0.6, x: 0.4, y: 0.4, width: 0.2, height: 0.2 },
  ],
  message: null,
}

describe('DetectionPanel', () => {
  it('summarizes groups and highest confidence', () => {
    render(<DetectionPanel overlay={overlay} staleMs={500} detectionStatus="active" />)

    expect(screen.getByText('Total')).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    expect(screen.getByText('2 / 90% max')).toBeInTheDocument()
    expect(screen.getByText('1 / 70% max')).toBeInTheDocument()
    expect(screen.getByText('1 / 60% max')).toBeInTheDocument()
  })

  it('shows no-objects message for active empty detections', () => {
    render(<DetectionPanel overlay={{ ...overlay, detections: [] }} staleMs={500} detectionStatus="active" />)

    expect(screen.getByText('No fish, feed, or debris detected')).toBeInTheDocument()
  })

  it('shows unavailable safe message', () => {
    render(<DetectionPanel overlay={null} staleMs={500} detectionStatus="unavailable" statusMessage="Detection model unavailable" />)

    expect(screen.getByText('Detection model unavailable')).toBeInTheDocument()
  })
})
