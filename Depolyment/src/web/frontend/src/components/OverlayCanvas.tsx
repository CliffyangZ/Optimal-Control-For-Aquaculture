import { useEffect, useRef } from 'react'
import { DetectionsMessage } from '../types'
import styles from './OverlayCanvas.module.css'

const CLASS_COLORS: Record<string, string> = {
  fish: '#00ff00',
  feed: '#ffa500',
  lumpfish: '#0066ff',
  debris: '#9933ff',
}

interface OverlayCanvasProps {
  width: number
  height: number
  detections: DetectionsMessage | null
}

export function OverlayCanvas({ width, height, detections }: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height

    ctx.clearRect(0, 0, width, height)

    if (!detections || !detections.detections.length) return

    detections.detections.forEach(det => {
      const color = CLASS_COLORS[det.class_name] || '#ffffff'

      const x = det.x * width
      const y = det.y * height
      const w = det.width * width
      const h = det.height * height

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)

      const text = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`
      ctx.font = '12px sans-serif'
      ctx.fillStyle = color
      ctx.fillText(text, x, y - 5)
    })
  }, [width, height, detections])

  return (
    <canvas
      ref={canvasRef}
      className={styles.canvas}
      style={{ width: '100%', height: '100%' }}
    />
  )
}
