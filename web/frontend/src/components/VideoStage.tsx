import { forwardRef } from 'react'
import type { WebRtcState } from '../hooks/useWebRtcPlayer'
import type { StreamStatus } from '../types/streaming'

interface VideoStageProps {
  rtcState: WebRtcState
  streamStatus: StreamStatus
  children?: React.ReactNode
}

export const VideoStage = forwardRef<HTMLVideoElement, VideoStageProps>(
  function VideoStage({ rtcState, children }, ref) {
    const showVideo = rtcState === 'live' || rtcState === 'reconnecting'

    return (
      <div className="video-stage">
        {/* 尚未開始 */}
        {(rtcState === 'idle' || rtcState === 'stopped') && (
          <div className="video-message video-message--idle">
            <p>Press <strong>Start</strong> to begin monitoring.</p>
          </div>
        )}

        {/* 正在連線 */}
        {rtcState === 'connecting' && (
          <div className="video-message video-message--connecting">
            <div className="spinner" />
            <p>Connecting to stream…</p>
          </div>
        )}

        {/* 瀏覽器不支援 WebRTC */}
        {rtcState === 'unsupported' && (
          <div className="video-message video-message--unsupported">
            <p>
              WebRTC is not supported in this browser.
              Please use <strong>Chrome</strong> or <strong>Edge</strong>.
            </p>
          </div>
        )}

        {/* 連線錯誤（最終失敗） */}
        {rtcState === 'error' && (
          <div className="video-message video-message--error">
            <p>Unable to connect to the live stream.</p>
            <p className="video-message__hint">Check that MediaMTX is running and a camera is publishing.</p>
          </div>
        )}

        {/* 影片元素（live 時可見；reconnecting 時保留以利恢復） */}
        <video
          ref={ref}
          className="video-element"
          style={{ display: showVideo ? 'block' : 'none' }}
          autoPlay
          playsInline
          muted
        />

        {/* 重新連線 overlay（蓋在影片上方） */}
        {rtcState === 'reconnecting' && (
          <div className="video-overlay video-overlay--reconnecting">
            <div className="spinner" />
            <p>Reconnecting…</p>
          </div>
        )}

        {showVideo && children}
      </div>
    )
  }
)
