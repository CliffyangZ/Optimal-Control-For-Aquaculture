import { AnalyticsMessage } from '../types'
import styles from './HungerBadge.module.css'

const HUNGER_COLORS: Record<string, string> = {
  Mild: '#2ecc71',
  Moderate: '#f39c12',
  Severe: '#e74c3c',
}

interface HungerBadgeProps {
  analytics: AnalyticsMessage | null
}

export function HungerBadge({ analytics }: HungerBadgeProps) {
  const level = analytics?.hunger_level || 'Unknown'
  const color = HUNGER_COLORS[level] || '#aaa'
  const rate = analytics?.consumption_rate.toFixed(4) || '—'

  return (
    <div className={styles.badge} style={{ borderColor: color }}>
      <div className={styles.level} style={{ color }}>
        {level}
      </div>
      <div className={styles.rate}>
        Rate: {rate} log/frame
      </div>
    </div>
  )
}
