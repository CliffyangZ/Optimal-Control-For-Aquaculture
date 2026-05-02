# Aquaculture_CV_Project

## 1. System Architecture
![System Architecture](assets/Sys_arch.drawio.svg)

## 2. Software Architecture

![Software Architecture](assets/software_arch.drawio.svg)

### File Structure

```text
Aquaculture_CV_Project/
├── assets/
│   ├── Sys_arch.drawio.svg
│   └── software_arch.drawio.svg
├── config/
│   └── setting.py
├── core/
│   ├── __init__.py
│   ├── README.md
│   ├── activity_levels.py
│   ├── models.py
│   ├── ripple_detector.py
│   └── stream_health.py
├── drivers/
│   └── README.md
├── streaming/
│   ├── __init__.py
│   └── latest_frame_reader.py
├── validation/
│   ├── __init__.py
│   ├── clip_manifest.py
│   ├── clips.example.yml
│   ├── clips.local.yml
│   └── run_clip_validation.py
├── web/
│   ├── README.md
│   └── app.py
├── tests/
│   ├── integration/
│   └── unit/
├── specs/
│   └── 001-rtsp-ripple-monitoring/
├── doc/
│   └── hardware_spec/
├── Dockerfile.mediamtx
├── mediamtx.yml
├── pyproject.toml
├── develop.md
└── README.md
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

## 5. Local Stream Relay Setup (Low-Latency Ripple Monitoring)

The ripple monitoring feature requires a local RTSP relay (MediaMTX) to accept a smartphone camera stream and expose it to the application.

### MediaMTX Setup

1. Download MediaMTX from https://github.com/bluenviron/mediamtx/releases
2. Start the relay with the default configuration:
   ```bash
   ./mediamtx
   ```
3. The relay will expose the default RTSP path for publishing: `rtsp://127.0.0.1:8554/`

### MediaMTX Docker Setup

Build the local MediaMTX image with FFmpeg and Raspberry Pi camera support:

```bash
docker build -f Dockerfile.mediamtx -t aquaculture-mediamtx .
```

Run as an RTSP relay for a phone or FFmpeg publisher:

```bash
docker run --rm -it -p 8554:8554 aquaculture-mediamtx
```

Run on Raspberry Pi when using the Pi camera path `rtsp://<pi-host>:8554/cam`:

```bash
docker run --rm -it \
  --network=host \
  --privileged \
  --tmpfs /dev/shm:exec \
  -v /run/udev:/run/udev:ro \
  aquaculture-mediamtx
```

The Docker configuration is in `mediamtx.yml`. It exposes RTSP over TCP on port `8554`, accepts external publishers at `/drone_live`, and exposes the Raspberry Pi camera at `/cam` when available.

### Smartphone Stream Configuration

Configure your smartphone camera app to publish to: `rtsp://127.0.0.1:8554/drone_live`

The application will read from: `rtsp://127.0.0.1:8554/drone_live`

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
