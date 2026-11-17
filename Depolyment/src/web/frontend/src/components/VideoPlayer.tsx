import { useEffect, useRef } from 'react'
import styles from './VideoPlayer.module.css'

interface VideoPlayerProps {
  whepUrl: string | null
}

export function VideoPlayer({ whepUrl }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const pcRef = useRef<RTCPeerConnection | null>(null)

  useEffect(() => {
    if (!whepUrl) return

    const setupWhep = async () => {
      try {
        const pc = new RTCPeerConnection()
        pcRef.current = pc

        pc.addTransceiver('video', { direction: 'recvonly' })

        pc.ontrack = (event) => {
          if (videoRef.current) {
            videoRef.current.srcObject = event.streams[0]
          }
        }

        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        const response = await fetch(whepUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/sdp',
          },
          body: offer.sdp,
        })

        if (!response.ok) {
          throw new Error(`WHEP failed: ${response.statusText}`)
        }

        const answerSdp = await response.text()
        const answer = new RTCSessionDescription({
          type: 'answer',
          sdp: answerSdp,
        })
        await pc.setRemoteDescription(answer)
      } catch (err) {
        console.error('WHEP connection error:', err)
      }
    }

    setupWhep()

    return () => {
      if (pcRef.current) {
        pcRef.current.close()
        pcRef.current = null
      }
    }
  }, [whepUrl])

  return (
    <div className={styles.videoContainer}>
      <video
        ref={videoRef}
        className={styles.video}
        autoPlay
        playsInline
        controls={false}
      />
    </div>
  )
}
