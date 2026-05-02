# Quickstart: Low-Latency Ripple Monitoring

This MVP is for local development or trusted local-network use only. Do not expose the Streamlit interface or local RTSP relay to the public internet.

## Prerequisites

- Python 3.12.
- MediaMTX installed locally.
- A smartphone streaming app capable of publishing to the local relay.
- Laptop/workstation and phone on the same stable local network.
- Local validation clips stored outside git when running prepared clip validation.

## 1. Configure the Local Stream Relay

Start MediaMTX on the workstation using a local configuration that accepts a phone stream and exposes a local RTSP path such as:

```text
rtsp://127.0.0.1:8554/drone_live
```

The MVP application does not start MediaMTX automatically. If the relay is not running or the phone is not connected, the UI reports missing or interrupted input without distinguishing the exact external cause.

## 2. Start Phone Streaming

On the smartphone streaming app:

- Set the stream target to the workstation's MediaMTX publish URL.
- Prefer 720p output for MVP validation.
- Keep the phone and workstation on the same local network.
- Point the camera at the water surface.

## 3. Configure Monitoring

Set the local stream source and thresholds in `config/setting.py` according to the configuration contract:

```python
STREAM_URL = "rtsp://127.0.0.1:8554/drone_live"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_DISPLAY_FPS = 30
TARGET_DETECTION_FPS = 10
INTERRUPTION_TIMEOUT_SECONDS = 3
```

Default activity thresholds are:

- Low: 0-33
- Medium: 34-66
- High: 67-100

## 4. Launch the Monitoring Interface

Run the Streamlit application from the repository root:

```bash
streamlit run web/app.py
```

Opening the interface must show `Waiting` and must not start reading the stream.

## 5. Start and Stop Monitoring

- Select `Start` to begin reading the configured stream.
- Confirm the UI shows `Active` when frames are current.
- Confirm the single video pane defaults to processed output.
- Use the view-mode control to switch to raw live view when needed.
- Select `Stop` to stop stream reading and return to `Waiting`.

After Stop, previous frame, score, and activity level must be cleared or clearly marked inactive.

## 6. Observe Stream Health

During active monitoring, verify these states:

- `Active`: Frames are arriving and detection is usable.
- `Interrupted`: No new frame has arrived for three seconds.
- `Detection Unavailable`: Frames are current but the scene is too dark, too bright, unstable, or otherwise low confidence.
- `Waiting`: Monitoring has not started or has been stopped.

When interrupted, restore the phone stream or relay input. The UI should return to Active when new frames arrive from the same configured source without restarting the whole application.

## 7. Validate Prepared Clips

Create a local validation manifest such as `validation/clips.local.yml`. The manifest may be committed only if it does not include private machine-specific paths; video files should remain outside git.

The validation set should include at least:

- Three Low activity clips.
- Three Medium activity clips.
- Three High activity clips.
- One low-confidence or interference clip.

Run the prepared clip validation command after implementation provides it:

```bash
python -m validation.run_clip_validation --manifest validation/clips.local.yml
```

The MVP target is at least 8 passing trials out of 10 for expected activity or low-confidence behavior.

## 8. Validate Live Latency

During a live phone-stream session:

- Confirm 720p visible updates target 30 FPS under normal local network conditions.
- Confirm ripple detection updates target at least 10 times per second when frames are usable.
- Confirm 95% of visible updates reflect camera scene changes within two seconds.
- Run for at least ten minutes and confirm visible delay does not accumulate by more than one second.
- Stop the phone stream and confirm interruption is reported within five seconds.

## Troubleshooting

- If the UI stays `Waiting`, select `Start`.
- If the UI becomes `Interrupted`, check that MediaMTX is running and the phone app is publishing to the expected path.
- If Detection Unavailable appears, improve lighting, reduce glare, reduce camera motion, or point the camera at a more stable water surface.
- If latency grows over time, verify the implementation is using latest-frame-only access and not replaying queued frames.
