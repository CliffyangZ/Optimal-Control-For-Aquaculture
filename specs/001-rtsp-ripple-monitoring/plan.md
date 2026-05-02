# Implementation Plan: Low-Latency Ripple Monitoring

**Branch**: `001-rtsp-ripple-monitoring` | **Date**: 2026-05-02 | **Spec**: `specs/001-rtsp-ripple-monitoring/spec.md`
**Input**: Feature specification from `specs/001-rtsp-ripple-monitoring/spec.md`

## Summary

Build a local, trusted-network MVP that lets an aquaculture operator start and stop monitoring of a smartphone camera stream routed through a manually started MediaMTX RTSP relay. The implementation will use a latest-frame-only streaming reader, OpenCV ripple scoring with fixed thresholds, and a Streamlit interface that displays processed or raw 720p output, stream health, frame age, detection age, activity level, and activity score without accumulating playback delay.

## Technical Context

**Language/Version**: Python 3.12, matching `pyproject.toml` `requires-python >=3.12`  
**Primary Dependencies**: OpenCV (`cv2`) for RTSP frame ingestion and ripple analysis; Streamlit for local monitoring UI; MediaMTX as an external manually started local stream relay; pytest for automated checks  
**Storage**: Configuration files and local validation manifest only; no database, no committed video binaries, no product image/video recording  
**Testing**: pytest unit/integration tests for configuration, latest-frame buffering, health transitions, and detection scoring; manual validation with MediaMTX, phone streaming, and local validation clips  
**Target Platform**: Operator laptop/workstation on a trusted local network, running the stream relay and Streamlit interface locally  
**Project Type**: Local Python Streamlit application with separate configuration, streaming, computer vision, validation, and UI modules  
**Performance Goals**: Target 720p visible monitoring at 30 FPS; at least 10 ripple detection updates per second when usable frames are available; 95% of visible updates within 2 seconds of camera scene changes; no more than 1 second additional delay accumulation over a 10-minute session  
**Constraints**: Start button must be required before stream reading; Stop must return to Waiting and mark previous output inactive; three seconds without a frame means Interrupted; detection must withhold score and activity level for low-confidence scenes; latest frame must be preferred over older queued frames; MVP is local/trusted-network only with no login or RBAC  
**Scale/Scope**: Single operator session, one configured smartphone RTSP source, one local relay, one displayed video pane with raw/processed toggle, fixed threshold activity mapping, prepared local validation clips referenced by manifest

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Spec-Driven Development**: PASS. The plan traces to `specs/001-rtsp-ripple-monitoring/spec.md`, especially FR-001 through FR-034 and SC-001 through SC-015.
- **English Artifacts**: PASS. New plan, research, contract, data model, quickstart, code comments, and task notes are required to be English.
- **Module Boundaries**: PASS. Ownership is separated into `config/` for settings, `streaming/` for stream acquisition and latest-frame access, `core/` for OpenCV ripple analysis and activity mapping, `web/` for Streamlit UI, and `validation/` for clip-manifest validation tooling. `drivers/` remains reserved for hardware drivers and is not used for the MVP stream relay.
- **Simplicity**: PASS. The MVP uses direct Python modules, a latest-frame holder, fixed thresholds, and no application state manager, database, trained model, authentication layer, actuator placeholder, recording pipeline, or generic plugin abstraction.
- **Verification**: PASS. Verification combines pytest coverage for module behavior, prepared local clip validation for activity behavior, and manual live phone-stream validation for latency, Start/Stop, and health states.

## Project Structure

### Documentation (this feature)

```text
specs/001-rtsp-ripple-monitoring/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── configuration.md
│   ├── monitoring-ui.md
│   └── validation-manifest.md
└── tasks.md
```

### Source Code (repository root)

```text
config/
└── setting.py              # Stream URL, thresholds, target FPS, interruption timeout, validation paths

streaming/
├── __init__.py
└── latest_frame_reader.py  # RTSP connection, latest-frame-only access, dropped stale frames, health timestamps

core/
├── __init__.py
├── ripple_detector.py      # OpenCV ripple scoring, confidence checks, visual marker generation
└── activity_levels.py      # Fixed 0-100 score thresholds mapped to Low/Medium/High

web/
├── app.py                  # Streamlit Start/Stop controls, raw/processed toggle, status and ages
└── components.py           # Small UI rendering helpers only if app.py becomes unclear

validation/
├── __init__.py
├── clip_manifest.py        # Local manifest loading and validation, no video binaries committed
└── run_clip_validation.py  # Prepared clip validation command

tests/
├── unit/
│   ├── test_activity_levels.py
│   ├── test_ripple_detector.py
│   └── test_stream_health.py
└── integration/
    ├── test_latest_frame_reader.py
    └── test_validation_manifest.py
```

**Structure Decision**: Use the repository's existing top-level module style and add only the missing ownership directories needed by the specification. `streaming/` owns live frame acquisition because FR-002, FR-006, FR-018, and MD-004 require latest-frame-only behavior and stream health separate from CV and UI. `core/` owns explainable OpenCV activity scoring. `web/` owns the Streamlit operator interface. `config/` owns thresholds and source parameters. `validation/` owns repeatable clip validation outside the product UI.

## Phase 0 Research Summary

Research output is captured in `specs/001-rtsp-ripple-monitoring/research.md`. All implementation choices are resolved and no open clarification markers remain.

## Phase 1 Design Summary

Design output is captured in:

- `specs/001-rtsp-ripple-monitoring/data-model.md`
- `specs/001-rtsp-ripple-monitoring/contracts/configuration.md`
- `specs/001-rtsp-ripple-monitoring/contracts/monitoring-ui.md`
- `specs/001-rtsp-ripple-monitoring/contracts/validation-manifest.md`
- `specs/001-rtsp-ripple-monitoring/quickstart.md`

## Post-Design Constitution Check

- **Spec-Driven Development**: PASS. Data entities, contracts, and quickstart map directly to feature requirements and success criteria.
- **English Artifacts**: PASS. Phase 0 and Phase 1 artifacts are English.
- **Module Boundaries**: PASS. Contracts describe stable data exchanged across configuration, streaming, CV, validation, and UI boundaries without mixing ownership.
- **Simplicity**: PASS. The design keeps fixed thresholds, local files, and direct module contracts; no speculative extension points are introduced.
- **Verification**: PASS. Quickstart and contracts define automated and manual checks for low-latency monitoring, stream interruption, detection unavailable, and validation clip behavior.

## Complexity Tracking

No constitution violations.
