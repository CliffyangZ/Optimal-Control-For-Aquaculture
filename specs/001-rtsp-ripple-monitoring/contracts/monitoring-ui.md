# Contract: Monitoring UI

The monitoring UI is a local Streamlit interface owned by `web/`. User-facing text must be English.

## Initial State

- The interface opens in `Waiting`.
- Stream reading does not start until the operator selects `Start`.
- No previous score, activity level, or frame is presented as current.
- A local/trusted-network warning is visible in the quickstart and may be shown in the UI.

## Controls

- `Start`: Begins reading from the configured local stream source.
- `Stop`: Stops reading, releases stream resources, returns health to `Waiting`, and clears or marks prior output inactive.
- `View Mode`: Switches the single video display between `Processed` and `Raw`; default is `Processed`.

## Status Display

The UI must show exactly one current health state from this set:

- `Waiting`
- `Active`
- `Interrupted`
- `Detection Unavailable`

## Active Display Fields

When frames are current, the UI shows:

- Latest image in the selected view mode.
- Frame age.
- Detection age, if a detection result exists.
- Activity score on a 0-100 scale when detection is available.
- Activity level `Low`, `Medium`, or `High` when detection is available.
- Visual marker or processed view when detection is available.

## Detection Unavailable Behavior

When frames are current but detection confidence is insufficient, the UI shows:

- Latest image remains visible.
- Health state `Detection Unavailable`.
- Reason equivalent to `Low confidence scene`, including too dark, too bright, or unstable scene where applicable.
- No activity score.
- No activity level.

## Interrupted Behavior

When no new image has arrived for three seconds during active monitoring, the UI shows:

- Health state `Interrupted`.
- No stale image or detection result labeled as current.
- Continued attempt to read from the same configured source until Stop or frames return.

## Stop Behavior

Within five seconds after Stop, the UI must show:

- Health state `Waiting`.
- Stream reading stopped.
- Previous frame, score, and activity level cleared or marked inactive.

## Non-Goals

- No login or role-based access control.
- No fish-location estimate.
- No feeding recommendation.
- No actuator controls or placeholders.
- No automatic image saving or video recording.
