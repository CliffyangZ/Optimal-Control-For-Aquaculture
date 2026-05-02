# Phase 0 Research: Low-Latency Ripple Monitoring

## Decision: Use Python 3.12 with OpenCV and Streamlit

**Rationale**: The repository already targets Python 3.12, and the project constitution identifies OpenCV processing and Streamlit UI as project constraints. OpenCV supports RTSP frame ingestion and simple image-processing algorithms suitable for the MVP's explainable, no-trained-model requirement. Streamlit supports a fast local operator interface with Start/Stop controls and status panels.

**Alternatives considered**: A browser-only WebRTC app would improve browser-native video handling but adds frontend/backend complexity not required by the MVP. A native desktop UI would add packaging work without improving validation. A trained ML model was rejected because FR-032 requires an explainable low-latency approach without trained model data.

## Decision: Use MediaMTX as a manual external RTSP relay

**Rationale**: The feature requires a smartphone stream routed through a required local stream relay. MediaMTX is lightweight, local, and well-suited to accepting phone RTSP/RTMP publishing while exposing a stable local RTSP URL to OpenCV. Keeping relay setup manual satisfies the MVP quickstart requirement without mixing external process management into the product UI.

**Alternatives considered**: Direct phone-to-OpenCV capture is less stable and makes phone app details harder to document. Embedding relay startup into the app was rejected because the MVP requires manual relay setup and simple module boundaries. Cloud relay or public streaming services were rejected because the MVP is local/trusted-network only.

## Decision: Use latest-frame-only acquisition with no replay queue

**Rationale**: FR-002, FR-006, FR-007, and SC-002 require the displayed view not to fall progressively behind when processing cannot keep up. A dedicated acquisition thread that continuously replaces a single latest-frame slot prevents queue buildup. Consumers read the newest available frame and tolerate detection results updating at a lower rate than video display.

**Alternatives considered**: A FIFO frame queue preserves every frame but accumulates latency when analysis is slower than input. Processing synchronously inside the UI loop is simpler but risks Streamlit refresh delays and stale images. Multiple buffering layers were rejected as unnecessary for the single-source MVP.

## Decision: Separate visible frame rate from detection update rate

**Rationale**: FR-011 targets 30 visible FPS while FR-012 targets at least 10 detection updates per second. Treating display and analysis as separate consumers lets the UI remain current even when detection is slower, while still showing detection age so the operator can recognize stale analysis.

**Alternatives considered**: Requiring every displayed frame to have fresh detection would reduce visible frame rate and increase latency. Reusing detection without showing age was rejected because FR-024 requires frame and detection age visibility.

## Decision: Implement explainable OpenCV ripple scoring with fixed thresholds

**Rationale**: The MVP must detect visible ripple activity without trained data and map scores to Low, Medium, and High using fixed thresholds. An OpenCV pipeline based on grayscale conversion, blur/noise suppression, frame difference or optical-motion cues, edge/texture response, and region marking can produce a normalized 0-100 activity score with clear failure modes for poor lighting or unstable scenes.

**Alternatives considered**: YOLO or segmentation models need labeled data and are outside FR-032. Full physical wave modeling is too complex for the MVP. Manual operator labeling alone does not satisfy the live detection requirements.

## Decision: Treat low-confidence scenes as Detection Unavailable

**Rationale**: FR-023, FR-030, FR-033, and SC-015 require withholding activity score and level when scenes are too dark, too bright, unstable, or otherwise unreliable. The detector should return a status and reason instead of forcing Low/Medium/High in low-confidence conditions, while the latest frame remains visible if stream frames are current.

**Alternatives considered**: Mapping unreliable scenes to Low was rejected because it hides detection failure. Showing score with a warning was rejected because the specification requires score and level to be withheld until detection is reliable again.

## Decision: Use local manifest-based validation clips outside git

**Rationale**: FR-026, FR-027, SC-007, and SC-012 require repeatable validation without committing video binaries. A manifest can list local file paths, expected activity categories, and interference labels while allowing each workstation to store clips locally or externally.

**Alternatives considered**: Committing sample videos was rejected due to repository size and FR-027. Hard-coded local paths were rejected because they are not repeatable across machines. Treating validation clips as product input mode was rejected because the MVP product flow is live monitoring.

## Decision: Document local/trusted-network operation with no authentication

**Rationale**: FR-028 and SC-013 explicitly limit MVP use to local or trusted local networks and do not require login or RBAC. The quickstart must warn operators not to expose the Streamlit interface beyond that environment.

**Alternatives considered**: Adding authentication now would increase scope and conflict with simplicity. Public network exposure was rejected for security and because the MVP does not include hardening requirements.
