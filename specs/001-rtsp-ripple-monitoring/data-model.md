# Data Model: Low-Latency Ripple Monitoring

## Live Stream Source

Represents the configured input exposed by the local stream relay.

**Fields**:

- `url`: RTSP URL used by the monitoring app, for example `rtsp://127.0.0.1:8554/drone_live`.
- `target_width`: Target capture/display width, default 1280.
- `target_height`: Target capture/display height, default 720.
- `target_display_fps`: Target visible update rate, default 30.
- `interruption_timeout_seconds`: Seconds without a new frame before stream health becomes Interrupted, default 3.

**Validation Rules**:

- `url` must be configured before Start can begin reading.
- Resolution defaults must target 720p.
- `interruption_timeout_seconds` must be 3 for the MVP unless the specification changes.

## Local Stream Relay

Represents the manually started external relay that receives the phone stream.

**Fields**:

- `relay_name`: Human-readable relay name, default `MediaMTX`.
- `publish_path`: Documented phone publishing path.
- `read_url`: Local RTSP URL consumed by the app.
- `startup_command`: Documentation-only command or location for starting the relay.

**Validation Rules**:

- The product UI does not distinguish inactive relay from inactive phone stream.
- Relay setup and troubleshooting belong in quickstart documentation.

## Current Frame

Represents the latest frame available for display or analysis.

**Fields**:

- `image`: Current image array in OpenCV format.
- `captured_at`: Monotonic or wall-clock timestamp when the app received the frame.
- `sequence_number`: Increasing local sequence number assigned by the frame reader.
- `source_width`: Width of the received frame.
- `source_height`: Height of the received frame.

**Relationships**:

- Belongs to one Monitoring Session.
- May be analyzed to produce one Ripple Detection Result.

**Validation Rules**:

- Only the newest frame is current; older unprocessed frames must be dropped.
- Frame age must be shown in the monitoring interface during active monitoring.

## Ripple Detection Result

Represents the most recent completed analysis result.

**Fields**:

- `status`: `available` or `unavailable`.
- `activity_score`: Integer 0-100 when status is `available`; absent when unavailable.
- `activity_level`: `Low`, `Medium`, or `High` when status is `available`; absent when unavailable.
- `confidence`: Numeric confidence or reliability indicator used internally for availability decisions.
- `reason`: Required when status is `unavailable`, for example `Low confidence scene`, `Too dark`, `Too bright`, or `Unstable scene`.
- `visual_marker`: Processed overlay or marker metadata for the displayed frame.
- `frame_sequence_number`: Sequence number of the frame used for analysis.
- `completed_at`: Timestamp when detection completed.

**Relationships**:

- Produced from one Current Frame.
- Displayed by one Monitoring Session until a newer result replaces it or Stop marks it inactive.

**Validation Rules**:

- Score must be on a 0-100 scale.
- Low is 0-33, Medium is 34-66, High is 67-100 by default.
- Detection Unavailable must withhold score and activity level.
- Detection age must be shown when a result exists during active monitoring.

## Stream Health State

Represents the operator-facing monitoring status.

**Values**:

- `Waiting`: Monitoring is inactive or the operator has not selected Start.
- `Active`: Monitoring is started and current frames are arriving.
- `Interrupted`: Monitoring is started but no new frame has arrived for three seconds.
- `Detection Unavailable`: Frames are current but reliable activity scoring is not available.

**State Transitions**:

- `Waiting` to `Active`: Operator selects Start and frames arrive.
- `Waiting` to `Interrupted`: Operator selects Start but no frames arrive within the expected window.
- `Active` to `Interrupted`: No new frame for three seconds.
- `Interrupted` to `Active`: New frames arrive from the same configured source.
- `Active` to `Detection Unavailable`: Frames are current but detection confidence is too low.
- `Detection Unavailable` to `Active`: Detection becomes reliable while frames remain current.
- Any active state to `Waiting`: Operator selects Stop.

**Validation Rules**:

- Stop clears or marks previous frame, score, and activity level inactive.
- Interrupted must not show stale output as current.

## Monitoring Session

Represents a continuous operator-controlled observation period.

**Fields**:

- `session_id`: Local identifier for the session.
- `started_at`: Timestamp when Start was selected.
- `stopped_at`: Timestamp when Stop was selected, if stopped.
- `health_state`: Current Stream Health State.
- `view_mode`: `processed` or `raw`, default `processed`.
- `latest_frame`: Current Frame if active and available.
- `latest_detection`: Latest Ripple Detection Result if available and active.

**Validation Rules**:

- Opening the interface must not create an active reading session.
- One configured source is monitored per MVP session.
- Sessions must support at least ten minutes without accumulating visible delay under normal local network conditions.

## Validation Clip

Represents a local prepared clip used outside the product UI.

**Fields**:

- `id`: Unique manifest identifier.
- `path`: Local file path, not committed video content.
- `expected_activity`: `Low`, `Medium`, `High`, or `Detection Unavailable`.
- `interference_labels`: Optional labels such as `glare`, `foam`, `rain`, `too_dark`, `too_bright`, or `unstable`.
- `notes`: Optional validation notes.

**Validation Rules**:

- MVP validation set must include at least three Low, three Medium, three High, and one interference clip.
- Missing clip files must be reported by validation tooling without requiring files in git.

## Validation Manifest

Represents the repeatable list of local validation clips.

**Fields**:

- `version`: Manifest schema version.
- `clips`: List of Validation Clip entries.
- `created_for`: Feature or validation target name.

**Validation Rules**:

- Paths may be absolute or relative to the manifest file.
- Manifest must be valid before prepared clip validation runs.
