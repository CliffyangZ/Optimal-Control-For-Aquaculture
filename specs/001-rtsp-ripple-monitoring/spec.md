# Feature Specification: Low-Latency Ripple Monitoring

**Feature Branch**: `001-rtsp-ripple-monitoring`  
**Created**: 2026-05-02  
**Status**: Draft  
**Input**: User description: "@develop.md"

<!-- Specifications MUST be written in English. User-facing text may use another
language only when the feature explicitly requires it. -->

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Monitor Live Water Ripples (Priority: P1)

An aquaculture operator wants to view a live camera feed from a drone-like or
mobile camera perspective and see whether water ripple activity is being detected
with low visible delay, so they can judge fish activity while the camera is
streaming.

**Why this priority**: This is the MVP capability. Without reliable live viewing
and ripple detection, the system cannot support operational observation.

**Independent Test**: Start a live camera stream pointed at water, open the
monitoring interface, and verify that the displayed view and ripple indicators
update continuously at 720p with a target visible update rate of 30 frames per
second and no more than two seconds of visible delay during normal local network
conditions.

**Acceptance Scenarios**:

1. **Given** a valid live camera stream is available, **When** the operator opens
   the monitoring interface, **Then** the operator sees the latest camera view and
   current ripple detection result.
2. **Given** visible ripple movement appears in the camera view, **When** the
   system processes the stream, **Then** the monitoring interface highlights or
   reports ripple activity for the current scene.
3. **Given** the stream continues for at least ten minutes, **When** the operator
   watches the monitoring interface, **Then** the displayed view remains current
   and does not fall progressively further behind the source stream.
4. **Given** the camera produces images faster than ripple analysis can complete,
   **When** monitoring continues, **Then** the system displays the latest
   available image and keeps using the latest completed detection result without
   replaying older images.
5. **Given** the operator has opened the monitoring interface, **When** the
   operator has not selected Start, **Then** the system remains in Waiting and
   does not read from the live stream.
6. **Given** monitoring is active, **When** the operator selects Stop, **Then**
   stream reading stops, the interface returns to Waiting, and current activity
   output is cleared or marked as inactive.

---

### User Story 2 - Observe Stream Health (Priority: P2)

An aquaculture operator wants clear stream health feedback so they can tell
whether monitoring is active, delayed, disconnected, or waiting for video input.

**Why this priority**: Field use requires quick diagnosis when the camera,
network, or relay is not producing usable images.

**Independent Test**: Start monitoring with a valid stream, interrupt the stream,
and restore it. Verify that the interface reports the active, disconnected, and
returned-to-active states in a way an operator can understand.

**Acceptance Scenarios**:

1. **Given** the camera stream is unavailable, **When** the operator opens the
   monitoring interface, **Then** the system reports that no live feed is
   available and does not show stale results as current.
2. **Given** a stream was active and then stops, **When** no new image is received
   within the expected refresh window, **Then** the interface reports a stream
   interruption.
3. **Given** a stream interruption has been reported, **When** the live stream
   resumes, **Then** the interface returns to active monitoring without requiring
   the operator to restart the whole application.
4. **Given** no new image has arrived for three seconds, **When** the operator
   views the monitoring interface, **Then** the stream health state is shown as
   interrupted.
5. **Given** monitoring is interrupted and the operator has not selected Stop,
   **When** new images become available from the same source, **Then** the
   interface returns to Active without requiring a full application restart.

---

### User Story 3 - Review Detection Data During Monitoring (Priority: P3)

An aquaculture operator wants concise detection data alongside the live view so
they can understand whether ripple activity is increasing, decreasing, or stable
during observation.

**Why this priority**: Detection data improves operational value after the live
monitoring loop is reliable.

**Independent Test**: Run monitoring against scenes with low and high ripple
activity and verify that the displayed data changes in a way that matches the
observed scene differences.

**Acceptance Scenarios**:

1. **Given** the monitoring interface is receiving live images, **When** ripple
   activity changes visibly, **Then** the displayed detection data updates within
   the same monitoring session.
2. **Given** low ripple activity is present, **When** the operator views the
   detection data, **Then** the data communicates a lower activity level than a
   visibly active ripple scene.
3. **Given** the operator observes the monitoring interface, **When** detection
   data is unavailable for the current image, **Then** the interface clearly
   distinguishes missing data from low activity.
4. **Given** glare, foam, rain, or camera motion makes the scene unreliable,
   **When** the system cannot confidently classify ripple activity, **Then** the
   interface shows a low-confidence or detection-unavailable state instead of a
   normal activity level.
5. **Given** detection is unavailable but current frames are still arriving,
   **When** the operator views the monitoring interface, **Then** the latest image
   remains visible while activity score and activity level are withheld.

---

### Edge Cases

- The live stream is not started when monitoring begins.
- The live stream stops temporarily and later resumes.
- Incoming images arrive slower than the monitoring refresh rate.
- The source camera sends frames faster than the system can analyze them.
- The source image contains glare, foam, rain, or camera motion that can resemble
  ripple activity.
- The camera resolution is higher than needed for reliable observation.
- The operator leaves monitoring active for an extended session.
- The local stream relay is not running before monitoring starts.
- The stream relay is running but no phone stream is currently connected.
- Validation clips are missing, incomplete, or not organized by expected activity
  category.
- The monitoring interface is accidentally exposed beyond the trusted local
  environment.
- Lighting is too dark, too bright, or the camera view is too unstable for
  reliable detection.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST let an operator monitor a live water-surface camera
  feed from a smartphone stream routed through a required local stream relay.
- **FR-002**: The system MUST display the latest available image instead of
  replaying accumulated older images when processing cannot keep up with the
  source stream.
- **FR-003**: The system MUST detect visible water ripple activity in each usable
  current image and present the result to the operator.
- **FR-004**: The system MUST present a processed visual view, detection status,
  or detection data that helps the operator understand current ripple activity.
- **FR-005**: The system MUST report stream health using the states Waiting,
  Active, Interrupted, and Detection Unavailable.
- **FR-006**: The system MUST avoid presenting stale images or stale detection
  results as current monitoring output.
- **FR-007**: The system MUST allow a monitoring session to continue for at least
  ten minutes without delay accumulating over time under normal local network
  conditions.
- **FR-008**: The system MUST support operator-readable status for active
  monitoring, waiting for input, interrupted stream, and detection unavailable.
- **FR-009**: The system MUST keep stream acquisition, ripple analysis,
  configuration, and monitoring display responsibilities separately described for
  planning and validation.
- **FR-010**: The system MUST provide a clear configuration file for the live
  stream source, activity thresholds, and target observation parameters without
  requiring UI editing.
- **FR-011**: The system MUST target 720p live monitoring with a visible update
  rate of 30 frames per second under normal local network conditions.
- **FR-012**: The system MUST allow ripple detection results to update at a lower
  rate than the displayed live image, with a target of at least 10 detection
  updates per second under normal local network conditions.
- **FR-013**: The system MUST present ripple activity as an activity level,
  numeric score, and visual marker for the current monitoring session.
- **FR-014**: The system MUST use configurable fixed thresholds to map ripple
  scores to activity levels for the MVP.
- **FR-015**: The system MUST not provide fish-location estimates, feeding
  recommendations, actuator control, actuator placeholders, feeding-control
  placeholders, automatic image saving, or video recording as part of this MVP.
- **FR-016**: The monitoring interface MUST use English user-facing text.
- **FR-017**: The monitoring interface MUST provide a single video display that
  defaults to processed output and allows switching to the raw live view.
- **FR-018**: The system MUST treat three seconds without a new image as an
  interrupted stream.
- **FR-019**: The monitoring interface MUST provide Start and Stop controls;
  opening the interface alone MUST NOT start stream reading.
- **FR-020**: After an interruption, the system MUST continue attempting to read
  the same configured source until the operator selects Stop or new images return.
- **FR-021**: The activity score MUST be presented on a 0 to 100 scale.
- **FR-022**: The default activity thresholds MUST be 0-33 for Low, 34-66 for
  Medium, and 67-100 for High until validation clips justify different values.
- **FR-023**: When detection is unavailable due to low-confidence scene content,
  the interface MUST show Detection Unavailable with a reason equivalent to Low
  confidence scene.
- **FR-024**: The interface MUST show the age of the latest displayed frame and
  the age of the latest completed detection result.
- **FR-025**: The feature MUST include a quickstart that explains the manual
  stream relay setup, phone streaming step, monitoring interface launch,
  Start/Stop controls, and validation flow.
- **FR-026**: The feature MUST support two validation paths: prepared clip
  validation for activity detection behavior and live phone-stream validation for
  low-latency monitoring behavior.
- **FR-027**: Validation video clips MUST NOT be required to be stored in the git
  repository; the project MUST define a local clip location or manifest format
  for repeatable validation.
- **FR-028**: The MVP MUST be documented for local or trusted local-network use
  only and MUST NOT require login or role-based access control.
- **FR-029**: Selecting Stop MUST return the interface to Waiting and clear or
  clearly mark any previous frame, score, or activity level as inactive.
- **FR-030**: Detection Unavailable MUST still allow the latest frame to remain
  visible when stream frames are current, but MUST withhold activity score and
  activity level until detection is reliable again.
- **FR-031**: The monitoring interface MUST NOT distinguish whether missing
  images are caused by an inactive relay or an inactive phone stream; the
  quickstart MUST provide troubleshooting steps for those external conditions.
- **FR-032**: The MVP detection approach MUST be suitable for low-latency,
  explainable monitoring without requiring trained model data.
- **FR-033**: Detection Unavailable MUST cover, at minimum, scenes that are too
  dark, too bright, or too unstable for reliable activity scoring.
- **FR-034**: Other interference sources such as glare, foam, rain, and water
  surface artifacts MAY be included in validation observations, but the MVP MUST
  NOT claim complete handling of every interference type.

### Module and Documentation Constraints *(mandatory)*

- **MD-001**: Feature scope MUST identify the owning module(s) and avoid mixing
  domain logic, computer vision processing, hardware drivers, configuration, and
  user interface responsibilities.
- **MD-002**: Any required code comments, docstrings, plan notes, and task notes
  MUST be written in English and explain intent or non-obvious behavior.
- **MD-003**: Feature scope MUST avoid speculative abstractions or unused
  extension points unless a requirement makes them necessary.
- **MD-004**: The planning model MUST keep live stream acquisition in a dedicated
  streaming responsibility that only manages stream connection, latest-frame
  access, dropped older frames, and stream health.
- **MD-005**: The planning model MUST not introduce a separate application state
  manager for this MVP unless a later requirement adds feeding workflow or
  broader business-state coordination.

### Key Entities *(include if feature involves data)*

- **Live Stream Source**: The configured source that provides the camera view for
  monitoring, supplied by a smartphone stream through the local stream relay.
- **Local Stream Relay**: The required external relay that receives the phone
  stream and exposes a stable local stream input for monitoring. It is started
  manually for the MVP.
- **Current Frame**: The latest usable image available for analysis and display;
  older unprocessed images are not treated as current.
- **Ripple Detection Result**: The current classification, score, region, or
  status that describes observed ripple activity for the current frame.
- **Activity Score**: A normalized 0 to 100 value that represents the current
  estimated ripple activity level for operator display.
- **Stream Health State**: The operator-facing status that indicates whether the
  stream is Waiting, Active, Interrupted, or Detection Unavailable.
- **Monitoring Session**: A continuous observation period that combines live view,
  detection results, and health status for an operator.
- **Validation Clip**: A prepared test clip with an expected activity category,
  used to validate threshold settings and detection behavior outside the product
  UI.
- **Validation Manifest**: A lightweight record that lists local validation clips,
  expected categories, and any interference labels needed to repeat MVP
  validation without committing video files.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: During normal local network conditions, 95% of operator-visible
  updates at 720p show live scene changes within two seconds of occurring at the
  camera while targeting 30 visible frames per second.
- **SC-002**: In a ten-minute monitoring session, observed display delay does not
  increase by more than one second from the beginning to the end of the session.
- **SC-003**: Operators can identify whether monitoring is active, waiting for
  input, interrupted, or detection is unavailable within five seconds of viewing
  the interface.
- **SC-004**: In validation scenes with visibly different ripple activity levels,
  the displayed detection result distinguishes low-activity and high-activity
  scenes in at least 8 out of 10 trials.
- **SC-005**: When the live stream is stopped during monitoring, the interface
  reports the interruption within five seconds and does not label the last image
  as current.
- **SC-006**: Ripple detection results update at least 10 times per second during
  95% of a normal local monitoring session when usable frames are available.
- **SC-007**: The MVP validation set includes at least three low-activity clips,
  three medium-activity clips, three high-activity clips, and one interference
  clip; the system meets expected activity or low-confidence behavior in at least
  8 out of 10 trials.
- **SC-008**: In the target validation environment, the phone and monitoring
  workstation are on the same stable local network, and the stream relay and
  monitoring interface run on the workstation.
- **SC-009**: Operators can start monitoring, stop monitoring, and confirm the
  Waiting state after stopping within five seconds of using the controls.
- **SC-010**: Operators can see both frame age and detection age during active
  monitoring and use them to identify stale frame or detection data.
- **SC-011**: A new operator can follow the quickstart to start the relay, stream
  from the phone, launch monitoring, use Start/Stop, and observe the four health
  states without undocumented steps.
- **SC-012**: The validation process can be completed with local validation clips
  referenced by a manifest or documented folder layout, without committing the
  clip files to the repository.
- **SC-013**: The quickstart warns operators not to expose the monitoring
  interface outside the local or trusted local-network environment.
- **SC-014**: After Stop is selected, operators can verify within five seconds
  that monitoring is inactive and no stale score is presented as current.
- **SC-015**: In validation scenes that are too dark, too bright, or visibly
  unstable, the system withholds activity score and activity level instead of
  presenting a normal Low, Medium, or High result.

## Assumptions

- The MVP is intended for local development or local network operation before
  field-hardening for drone deployment.
- A smartphone provides the MVP live water-surface view.
- The operator uses a laptop or workstation to view the monitoring interface.
- Initial validation focuses on water ripple observation and does not trigger
  automatic feeding or actuator control.
- Normal local network conditions mean the camera and monitoring device are on
  the same stable Wi-Fi or local network with adequate bandwidth for 720p video.
- The local stream relay is installed and started manually before monitoring.
- Test video clips may be used for validation and threshold calibration, but they
  are not exposed as a product input mode in the MVP interface.
- Validation clips are stored locally or externally and referenced by path or
  manifest; the repository does not need to contain video binaries.
- The first version uses visual ripple cues and relative activity indicators
  rather than calibrated biological feeding recommendations.
- The main product entry point is the web monitoring interface; any separate
  startup script is outside the primary operator flow for the MVP.
- The stream relay and monitoring interface run on the operator workstation for
  MVP validation.
- Authentication and authorization are out of scope for the MVP because operation
  is limited to local or trusted local-network environments.
- Hardware control and feeding workflow are out of scope for the MVP, including
  placeholder controls that could imply actuator readiness.
- The MVP does not need to diagnose every external stream setup failure from the
  UI; setup troubleshooting belongs in the quickstart.
