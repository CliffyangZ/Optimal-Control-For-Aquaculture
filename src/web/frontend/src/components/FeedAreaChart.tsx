import { useEffect, useRef } from 'react'
import { AnalyticsMessage } from '../types'
import styles from './FeedAreaChart.module.css'

interface FeedAreaChartProps {
  analytics: AnalyticsMessage | null
}

export function FeedAreaChart({ analytics }: FeedAreaChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !analytics || !analytics.frame_ids.length) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    const padding = 40
    const width = canvas.width - padding * 2
    const height = canvas.height - padding * 2

    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    ctx.strokeStyle = '#333'
    ctx.lineWidth = 1

    for (let i = 0; i <= 5; i++) {
      const y = padding + (height / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(canvas.width - padding, y)
      ctx.stroke()
    }

    const drawLine = (data: number[], color: string) => {
      if (data.length === 0) return

      const max = Math.max(...data, 1)
      const min = Math.min(...data, 0)
      const range = max - min || 1

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      data.forEach((value, index) => {
        const x = padding + (width / (data.length - 1 || 1)) * index
        const y = padding + height - ((value - min) / range) * height
        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })
      ctx.stroke()
    }

    drawLine(analytics.raw_area, '#b42828')
    drawLine(analytics.kalman_area, '#3498db')
    drawLine(analytics.ema_area, '#e67e22')

    ctx.fillStyle = '#666'
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText('Raw', padding + 5, padding + 15)

    ctx.fillStyle = '#3498db'
    ctx.fillText('Kalman', padding + 45, padding + 15)

    ctx.fillStyle = '#e67e22'
    ctx.fillText('EMA', padding + 115, padding + 15)
  }, [analytics])

  return (
    <canvas
      ref={canvasRef}
      className={styles.chart}
    />
  )
}
