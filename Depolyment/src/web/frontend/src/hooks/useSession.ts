import { useEffect, useRef } from 'react'

export function useSession() {
  const viewerIdRef = useRef<string>('')

  useEffect(() => {
    const startSession = async () => {
      try {
        const response = await fetch('/session/start', { method: 'POST' })
        if (!response.ok) throw new Error('Failed to start session')
        viewerIdRef.current = Math.random().toString(36).substring(7)
      } catch (err) {
        console.error('Session start error:', err)
      }
    }

    startSession()

    const heartbeatInterval = setInterval(() => {
      if (viewerIdRef.current) {
        fetch('/session/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ viewer_id: viewerIdRef.current }),
        }).catch(err => console.error('Heartbeat error:', err))
      }
    }, 15000)

    return () => {
      clearInterval(heartbeatInterval)
      if (viewerIdRef.current) {
        fetch('/session/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ viewer_id: viewerIdRef.current }),
        }).catch(err => console.error('Session stop error:', err))
      }
    }
  }, [])

  return { viewerId: viewerIdRef.current }
}
