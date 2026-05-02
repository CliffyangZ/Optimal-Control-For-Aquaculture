# Tasks: Low-Latency Ripple Monitoring

**Input**: Design documents from `specs/001-rtsp-ripple-monitoring/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Verification tasks are included because the constitution and implementation plan require pytest checks, prepared clip validation, and live phone-stream validation for camera ingestion and monitoring behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files and does not depend on incomplete tasks.
- **[Story]**: Maps the task to a user story from `specs/001-rtsp-ripple-monitoring/spec.md`.
- Every task includes an exact file path.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the Python module layout, dependencies, and shared documentation baseline.

- [ ] T001 Add OpenCV, Streamlit, pytest, NumPy, and PyYAML dependencies to `pyproject.toml`
- [ ] T002 Create package initialization files in `streaming/__init__.py`, `core/__init__.py`, `validation/__init__.py`, `tests/unit/__init__.py`, and `tests/integration/__init__.py`
- [ ] T003 [P] Create local validation clip ignore rules for `validation/clips.local.yml` and common video extensions in `.gitignore`
- [ ] T004 [P] Add a MediaMTX local relay setup note with the default RTSP path in `README.md`
- [ ] T005 [P] Normalize existing module README text to English in `core/README.md`, `drivers/README.md`, and `web/README.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement configuration, shared data shapes, and validation helpers required before any user story can be completed.

**CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T006 Implement MVP configuration constants from `contracts/configuration.md` in `config/setting.py`
- [ ] T007 [P] Implement activity threshold mapping and validation in `core/activity_levels.py`
- [ ] T008 [P] Implement shared dataclasses/enums for Current Frame, Ripple Detection Result, Stream Health State, and Monitoring Session in `core/models.py`
- [ ] T009 [P] Add unit tests for 0-100 activity thresholds and invalid threshold gaps in `tests/unit/test_activity_levels.py`
- [ ] T010 [P] Add unit tests for stream health state transitions and Stop returning to Waiting in `tests/unit/test_stream_health.py`
- [ ] T011 Implement validation manifest parsing, relative path resolution, duplicate ID checks, and required category counts in `validation/clip_manifest.py`
- [ ] T012 [P] Add integration tests for validation manifest parsing and missing clip reporting in `tests/integration/test_validation_manifest.py`
- [ ] T013 Confirm module boundary imports do not make `core/` depend on `web/` or `streaming/` in `tests/unit/test_module_boundaries.py`

**Checkpoint**: Foundation ready; user story implementation can now begin.

---

## Phase 3: User Story 1 - Monitor Live Water Ripples (Priority: P1) MVP

**Goal**: The operator can select Start, view the latest live 720p camera frame with processed ripple indication, avoid replaying older frames, and select Stop to return to Waiting.

**Independent Test**: Start a live phone stream through MediaMTX, open `web/app.py`, select Start, verify the displayed view and ripple indicator update at the latest available frame with no more than two seconds of visible delay, then select Stop and verify Waiting with inactive output.

### Tests for User Story 1

- [ ] T014 [P] [US1] Add unit tests for latest-frame replacement and stale queue avoidance in `tests/unit/test_latest_frame_buffer.py`
- [ ] T015 [P] [US1] Add unit tests for basic ripple detector available results and processed marker output in `tests/unit/test_ripple_detector.py`
- [ ] T016 [P] [US1] Add integration tests for reading mocked frame sequences without replaying older frames in `tests/integration/test_latest_frame_reader.py`
- [ ] T017 [P] [US1] Add UI contract test for initial Waiting state and Start/Stop labels in `tests/integration/test_monitoring_ui_contract.py`

### Implementation for User Story 1

- [ ] T018 [US1] Implement a thread-safe latest-frame buffer with sequence numbers and captured timestamps in `streaming/latest_frame_reader.py`
- [ ] T019 [US1] Implement RTSP start, continuous read, OpenCV buffer-size tuning, newest-frame access, and release-on-stop behavior in `streaming/latest_frame_reader.py`
- [ ] T020 [US1] Implement explainable OpenCV ripple scoring for usable frames in `core/ripple_detector.py`
- [ ] T021 [US1] Implement processed visual marker generation for current ripple activity in `core/ripple_detector.py`
- [ ] T022 [US1] Implement Streamlit session initialization that defaults to Waiting and does not read before Start in `web/app.py`
- [ ] T023 [US1] Implement Start and Stop controls wired to the frame reader and inactive-output clearing in `web/app.py`
- [ ] T024 [US1] Implement the single video pane with Processed default and Raw view toggle in `web/app.py`
- [ ] T025 [US1] Display current health, latest frame age, latest detection age, activity score, and activity level in `web/app.py`
- [ ] T026 [US1] Add live latency validation instructions for US1 to `specs/001-rtsp-ripple-monitoring/quickstart.md`
- [ ] T027 [US1] Run US1 pytest checks for latest-frame, detector, and UI contract behavior using `tests/unit/test_latest_frame_buffer.py`, `tests/unit/test_ripple_detector.py`, `tests/integration/test_latest_frame_reader.py`, and `tests/integration/test_monitoring_ui_contract.py`

**Checkpoint**: User Story 1 is independently functional and testable as the MVP.

---

## Phase 4: User Story 2 - Observe Stream Health (Priority: P2)

**Goal**: The operator can distinguish Waiting, Active, Interrupted, and returned-to-Active states during stream loss and recovery.

**Independent Test**: Start monitoring with a valid stream, interrupt the phone stream or relay input, verify Interrupted appears within five seconds and no stale result is current, restore the same source, and verify Active returns without restarting the application.

### Tests for User Story 2

- [ ] T028 [P] [US2] Add unit tests for three-second interruption timeout and recovery to Active in `tests/unit/test_stream_health.py`
- [ ] T029 [P] [US2] Add integration tests for simulated stream interruption and same-source recovery in `tests/integration/test_stream_recovery.py`
- [ ] T030 [P] [US2] Add UI contract tests for Interrupted messaging and stale-output suppression in `tests/integration/test_monitoring_ui_contract.py`

### Implementation for User Story 2

- [ ] T031 [US2] Add last-frame timestamp tracking and three-second interruption detection to `streaming/latest_frame_reader.py`
- [ ] T032 [US2] Add same-source reconnect or retry behavior after interruption until Stop is selected in `streaming/latest_frame_reader.py`
- [ ] T033 [US2] Implement health-state derivation for Waiting, Active, Interrupted, and Detection Unavailable in `core/stream_health.py`
- [ ] T034 [US2] Render operator-readable Waiting, Active, and Interrupted messages without relay-versus-phone diagnosis in `web/app.py`
- [ ] T035 [US2] Ensure Interrupted UI does not label the last frame or detection as current in `web/app.py`
- [ ] T036 [US2] Add stream interruption and recovery troubleshooting steps to `specs/001-rtsp-ripple-monitoring/quickstart.md`
- [ ] T037 [US2] Run US2 pytest checks for health transitions, recovery, and UI interruption behavior using `tests/unit/test_stream_health.py`, `tests/integration/test_stream_recovery.py`, and `tests/integration/test_monitoring_ui_contract.py`

**Checkpoint**: User Stories 1 and 2 are independently functional.

---

## Phase 5: User Story 3 - Review Detection Data During Monitoring (Priority: P3)

**Goal**: The operator can see concise activity score, activity level, visual marker, and Detection Unavailable reasons that distinguish real low activity from unreliable detection.

**Independent Test**: Run monitoring or prepared clips against low, medium, high, and unreliable scenes; verify score/level changes match visible ripple differences and that low-confidence scenes show Detection Unavailable while withholding score and level.

### Tests for User Story 3

- [ ] T038 [P] [US3] Add unit tests for Low, Medium, High activity level mapping and score bounds in `tests/unit/test_activity_levels.py`
- [ ] T039 [P] [US3] Add unit tests for too-dark, too-bright, and unstable Detection Unavailable reasons in `tests/unit/test_ripple_detector.py`
- [ ] T040 [P] [US3] Add integration tests for prepared validation clip result aggregation in `tests/integration/test_clip_validation_runner.py`
- [ ] T041 [P] [US3] Add UI contract tests for withholding score and activity level during Detection Unavailable in `tests/integration/test_monitoring_ui_contract.py`

### Implementation for User Story 3

- [ ] T042 [US3] Add low-confidence checks for too-dark, too-bright, and unstable scenes to `core/ripple_detector.py`
- [ ] T043 [US3] Ensure detector returns Detection Unavailable reason without activity score or activity level in `core/ripple_detector.py`
- [ ] T044 [US3] Render activity score, activity level, detection age, and Detection Unavailable reason according to `contracts/monitoring-ui.md` in `web/app.py`
- [ ] T045 [US3] Implement prepared clip validation runner with per-clip pass/fail and aggregate 8-of-10 reporting in `validation/run_clip_validation.py`
- [ ] T046 [US3] Add a documented sample manifest template without committed video paths in `validation/clips.example.yml`
- [ ] T047 [US3] Update validation instructions for low/medium/high/interference clip counts in `specs/001-rtsp-ripple-monitoring/quickstart.md`
- [ ] T048 [US3] Run US3 pytest checks for detector confidence, clip validation, and UI detection data using `tests/unit/test_activity_levels.py`, `tests/unit/test_ripple_detector.py`, `tests/integration/test_clip_validation_runner.py`, and `tests/integration/test_monitoring_ui_contract.py`

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation, and compliance checks across all stories.

- [ ] T049 [P] Update trusted local-network warning and non-goals in `README.md`
- [ ] T050 [P] Add manual MediaMTX and phone-stream validation checklist to `specs/001-rtsp-ripple-monitoring/quickstart.md`
- [ ] T051 Verify no actuator controls, feeding recommendations, fish-location estimates, automatic image saving, or video recording were introduced in `web/app.py`, `core/ripple_detector.py`, and `streaming/latest_frame_reader.py`
- [ ] T052 Run the full pytest suite for this feature using `tests/unit/` and `tests/integration/`
- [ ] T053 Run the prepared clip validation command documented in `specs/001-rtsp-ripple-monitoring/quickstart.md` against `validation/clips.local.yml`
- [ ] T054 Run the ten-minute live phone-stream validation and record the result in `specs/001-rtsp-ripple-monitoring/quickstart.md`
- [ ] T055 Review English-only comments, docstrings, task notes, and user-facing UI text in `config/`, `streaming/`, `core/`, `validation/`, `web/`, and `specs/001-rtsp-ripple-monitoring/`
- [ ] T056 Run constitution compliance review against `specs/001-rtsp-ripple-monitoring/spec.md`, `specs/001-rtsp-ripple-monitoring/plan.md`, and `specs/001-rtsp-ripple-monitoring/tasks.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational completion; delivers MVP live monitoring.
- **User Story 2 (Phase 4)**: Depends on Foundational completion and can integrate with US1 UI/reader files; independently testable through simulated interruption.
- **User Story 3 (Phase 5)**: Depends on Foundational completion and can integrate with US1 detector/UI files; independently testable through prepared clips and UI contract checks.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2; no dependency on US2 or US3.
- **User Story 2 (P2)**: Can start after Phase 2; uses the same health and stream abstractions but must remain testable without US3.
- **User Story 3 (P3)**: Can start after Phase 2; uses the same detector and UI abstractions but must remain testable without live stream recovery work.

### Within Each User Story

- Tests should be added before implementation where listed.
- Shared models and config precede services and UI integration.
- Streaming and detector logic precede UI display wiring.
- Manual validation should run at the story checkpoint before moving to the next story.

---

## Parallel Opportunities

- Setup tasks T003, T004, and T005 can run in parallel.
- Foundational tasks T007, T008, T009, T010, and T012 can run in parallel after T006 is understood.
- US1 test tasks T014, T015, T016, and T017 can run in parallel.
- US2 test tasks T028, T029, and T030 can run in parallel.
- US3 test tasks T038, T039, T040, and T041 can run in parallel.
- Documentation polish tasks T049 and T050 can run in parallel with code review tasks after story implementation.

---

## Parallel Example: User Story 1

```bash
Task: "Add unit tests for latest-frame replacement and stale queue avoidance in tests/unit/test_latest_frame_buffer.py"
Task: "Add unit tests for basic ripple detector available results and processed marker output in tests/unit/test_ripple_detector.py"
Task: "Add integration tests for reading mocked frame sequences without replaying older frames in tests/integration/test_latest_frame_reader.py"
Task: "Add UI contract test for initial Waiting state and Start/Stop labels in tests/integration/test_monitoring_ui_contract.py"
```

## Parallel Example: User Story 2

```bash
Task: "Add unit tests for three-second interruption timeout and recovery to Active in tests/unit/test_stream_health.py"
Task: "Add integration tests for simulated stream interruption and same-source recovery in tests/integration/test_stream_recovery.py"
Task: "Add UI contract tests for Interrupted messaging and stale-output suppression in tests/integration/test_monitoring_ui_contract.py"
```

## Parallel Example: User Story 3

```bash
Task: "Add unit tests for Low, Medium, High activity level mapping and score bounds in tests/unit/test_activity_levels.py"
Task: "Add unit tests for too-dark, too-bright, and unstable Detection Unavailable reasons in tests/unit/test_ripple_detector.py"
Task: "Add integration tests for prepared validation clip result aggregation in tests/integration/test_clip_validation_runner.py"
Task: "Add UI contract tests for withholding score and activity level during Detection Unavailable in tests/integration/test_monitoring_ui_contract.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Stop and validate live monitoring with MediaMTX, phone streaming, Start/Stop, latest-frame display, and processed/raw view switching.

### Incremental Delivery

1. Complete Setup and Foundational work.
2. Add US1 to deliver live ripple monitoring MVP.
3. Add US2 to make stream health operationally clear during interruption and recovery.
4. Add US3 to improve detection data, confidence handling, and prepared clip validation.
5. Complete Polish tasks for documentation, full verification, and constitution compliance.

### Validation Gates

1. US1 gate: 720p live view targets 30 FPS and no more than two seconds visible delay under normal local network conditions.
2. US2 gate: interrupted stream is reported within five seconds and recovers to Active without application restart.
3. US3 gate: validation scenes distinguish low and high activity in at least 8 out of 10 trials and low-confidence scenes withhold score and level.

---

## Notes

- Commit after each completed task or logical group.
- Do not add authentication, actuator controls, recording, fish-location estimates, feeding recommendations, or speculative extension points.
- Keep configuration values outside core algorithms.
- Keep code comments and docstrings English and focused on non-obvious intent.
