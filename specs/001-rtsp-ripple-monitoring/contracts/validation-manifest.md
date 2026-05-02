# Contract: Validation Manifest

Prepared clip validation uses a local manifest owned by `validation/`. The manifest references local video files without committing those files to git.

## Format

YAML is the preferred format for readability.

```yaml
version: 1
created_for: low-latency-ripple-monitoring
clips:
  - id: low-001
    path: /local/path/to/low-001.mp4
    expected_activity: Low
    interference_labels: []
    notes: Calm water reference clip
  - id: interference-001
    path: ./clips/unstable-camera.mp4
    expected_activity: Detection Unavailable
    interference_labels: [unstable]
    notes: Camera motion should withhold activity score
```

## Clip Fields

- `id`: Required unique string.
- `path`: Required local absolute path or path relative to the manifest file.
- `expected_activity`: Required value: `Low`, `Medium`, `High`, or `Detection Unavailable`.
- `interference_labels`: Optional list. Supported MVP labels include `glare`, `foam`, `rain`, `too_dark`, `too_bright`, and `unstable`.
- `notes`: Optional English text.

## Validation Rules

- At least three clips must be labeled `Low`.
- At least three clips must be labeled `Medium`.
- At least three clips must be labeled `High`.
- At least one clip must represent interference or low-confidence behavior.
- Missing files must fail prepared clip validation with a clear message.
- The manifest file may be committed; referenced video files must remain local or external.

## Validation Output

The validation command should report:

- Total clips checked.
- Missing clip paths.
- Per-clip predicted activity or Detection Unavailable result.
- Pass/fail against expected activity.
- Aggregate pass count, with the MVP target of at least 8 out of 10 trials passing.
