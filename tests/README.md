# Tests

## Summary (recorded 2026-05-03)

**100 tests passing** across contract, integration, and unit suites.

```
tests/contract/   – API and WebSocket payload contracts
tests/integration/ – Session lifecycle, overlay hub, multi-viewer, static serving
tests/unit/       – Stream health helpers, overlay payload normalization
```

## Running

```bash
uv run python -m pytest tests/contract/ tests/integration/ tests/unit/ -q
```

Ignore legacy Streamlit-era tests with:
```bash
uv run python -m pytest tests/ -q \
  --ignore=tests/integration/test_clip_validation_runner.py \
  --ignore=tests/integration/test_latest_frame_reader.py \
  --ignore=tests/integration/test_monitoring_ui_contract.py \
  --ignore=tests/integration/test_stream_recovery.py \
  --ignore=tests/integration/test_validation_manifest.py
```

## Notes

- RTSP/MediaMTX not required to run; WARN messages from OpenCV are expected when no stream is running.
- `pytest-asyncio` is configured in auto mode (`asyncio_mode = "auto"` in pyproject.toml).
