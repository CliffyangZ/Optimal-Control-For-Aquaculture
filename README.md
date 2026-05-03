# Aquaculture_CV_Project

## 1. System Architecture
![System Architecture](assets/Sys_Arch.drawio.svg)

## 2. Software Architecture

![Software Architecture](assets/software_arch.drawio.svg)

### File Structure

```text
Aquaculture_CV_Project/
├── config/
│   └── setting.py
├── core/
│   ├── activity_levels.py
│   ├── models.py
│   ├── ripple_detector.py
│   └── stream_health.py
├── drivers/
│   └── README.md
├── streaming/
│   └── latest_frame_reader.py
├── validation/
│   ├── clip_manifest.py
│   └── run_clip_validation.py
└── web/
    └── app.py
```

## 3. Development Principles

- Use Spec-Driven Development: write the English specification before planning or coding.
- Keep specifications, plans, tasks, code comments, and docstrings in English.
- Keep modules separated by responsibility: configuration, core logic, drivers, UI, and scripts.
- Prefer simple, direct implementations over speculative abstractions.
- Define verification steps before completing changes that affect vision, feeding, actuators, or safety.

## 4. Hardware Components

### The Brain
- MPU (Rasperry Pi Zero 2W)
- MCU (BMduino)
- FC (PixHawk)

### Drone Comopnents
- Camera (Rasperry Pi 5 IMX219 79.3 degree)
- Drone Basis (F450)

## 5. Local Stream Relay Setup (YOLOv26 Pose Estimation)

The pose estimation feature requires a local RTSP relay (MediaMTX) to accept a camera stream and expose it to the application.

### MediaMTX Setup Without Docker

1. Download MediaMTX from https://github.com/bluenviron/mediamtx/releases
2. Start the relay with the desktop configuration:
   ```bash
   ./mediamtx mediamtx.desktop.yml
   ```
3. The relay will expose the smartphone publishing path at `rtsp://<relay-host>:8554/drone_live`.

### MediaMTX Docker Setup: Windows, macOS, and Linux

Use the official generic MediaMTX image for desktop operating systems:

```bash
docker compose -f compose.mediamtx.desktop.yml up
```

This uses `mediamtx.desktop.yml`, which accepts external publishers at `/drone_live` and does not include Raspberry Pi camera-only settings.

### MediaMTX Docker Setup: Raspberry Pi Camera

Use the Raspberry Pi compose file only when the relay host is a Raspberry Pi and the `/cam` path should read from the Pi camera:

```bash
docker compose -f compose.mediamtx.rpi.yml up
```

This uses `mediamtx.rpi.yml`, exposes `/drone_live` for phone or FFmpeg publishers, and exposes `/cam` for the Raspberry Pi camera when available.

The legacy Raspberry Pi image can still be built manually if needed:

```bash
docker build -f Dockerfile.mediamtx -t aquaculture-mediamtx .
```

### MediaMTX Configuration Files

- `mediamtx.yml`: default desktop-compatible configuration.
- `mediamtx.desktop.yml`: explicit Windows, macOS, and Linux relay configuration.
- `mediamtx.rpi.yml`: Raspberry Pi configuration with `/cam` using `rpiCamera`.
- `compose.mediamtx.desktop.yml`: cross-platform desktop Docker Compose entrypoint.
- `compose.mediamtx.rpi.yml`: Raspberry Pi Docker Compose entrypoint with camera permissions.

### Smartphone Stream Configuration

Configure your smartphone camera app to publish to: `rtsp://<relay-host-lan-ip>:8554/drone_live`

For example, if the relay runs on a laptop at `192.168.200.101`, publish to:

```text
rtsp://192.168.200.101:8554/drone_live
```

The application will read from: `rtsp://127.0.0.1:8554/drone_live`

### YOLOv26 Pose Estimation Setup

1. Place the YOLOv26 pose model at `models/yolo-v26-pose.pt`, or set `POSE_YOLO_MODEL_PATH` to another local weights file.
2. Start the Streamlit app:
   ```bash
   streamlit run web/app.py
   ```

Optional thresholds can be tuned with `POSE_CONFIDENCE_THRESHOLD` and `POSE_KEYPOINT_CONFIDENCE_THRESHOLD`.

**⚠️ Local Trusted Network Only**

This is a local trusted-network setup. Do not expose the Streamlit interface or relay beyond your local network. There is no login, authentication, or role-based access control. This MVP is designed for operator use on a single workstation within a protected local environment.

### Non-Goals (Not Implemented)

The following features are explicitly NOT included in the MVP:

- **Actuator controls**: No automated water treatment, feeding, or device actuation.
- **Feeding recommendations**: No algorithmic feeding guidance.
- **Fish location estimates**: No tracking of fish position or count.
- **Automatic image saving**: Frames are not automatically saved to disk.
- **Video recording**: No recording of streams or detected activity.
- **Authentication or RBAC**: No login system or role-based access control.
- **Public network exposure**: MVP is designed for local networks only.
