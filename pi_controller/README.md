# Pi Controller

Raspberry Pi scheduled feeder controller. This package is intended to live as a separate submodule and run on the Raspberry Pi, while the laptop communicates with it over the local network.

It does not include a frontend. The laptop-side application should call the REST API and, if needed, subscribe to the WebSocket endpoint.

## Responsibilities

- Store feeding schedules locally in SQLite
- Run enabled schedules on the Raspberry Pi at matching weekday/time slots
- Trigger the feeder through either simulated mode or Raspberry Pi GPIO hardware mode
- Expose REST endpoints for laptop-side schedule management and manual feed control
- Expose WebSocket events for laptop-side status synchronization
- Keep feed run history for manual, scheduled, failed, cancelled, missed, and rejected runs

## Non-Responsibilities

- No React/Vite/frontend code in this package
- No browser UI serving from the Pi controller
- No public internet exposure or authentication layer
- No computer vision or monitoring UI logic

## Runtime

Install Python dependencies from the parent project or from the submodule environment:

```bash
uv sync --all-groups
```

Run in simulated mode for development:

```bash
PI_CONTROLLER_DRIVER_MODE=simulated uv run uvicorn pi_controller.app:app --host 0.0.0.0 --port 8001
```

Run in GPIO hardware mode on Raspberry Pi:

```bash
PI_CONTROLLER_DRIVER_MODE=hardware \
PI_CONTROLLER_GPIO_CHANNEL=17 \
PI_CONTROLLER_GPIO_ACTIVE_HIGH=true \
uv run uvicorn pi_controller.app:app --host 0.0.0.0 --port 8001
```

The laptop should connect to the Pi LAN address, for example `http://<pi-lan-ip>:8001/controller/status`.

## API Surface

- `GET /`: service descriptor for laptop discovery
- `GET /controller/config`: controller configuration and endpoint paths
- `GET /controller/status`: current feeder and scheduler state
- `GET /controller/snapshot`: config, status, schedules, and recent runs in one response
- `GET /controller/schedules`: list schedules
- `POST /controller/schedules`: create schedule
- `PUT /controller/schedules/{schedule_id}`: update schedule
- `PATCH /controller/schedules/{schedule_id}/enabled`: enable or disable schedule
- `DELETE /controller/schedules/{schedule_id}`: soft delete schedule
- `POST /controller/feed-runs/manual`: start manual feed
- `POST /controller/feed-runs/stop`: stop active feed
- `GET /controller/feed-runs`: recent feed run history
- `WS /ws/controller`: laptop-side synchronization channel

## Configuration

- `PI_CONTROLLER_ID`: controller identity, default `pi-controller-1`
- `PI_CONTROLLER_STORAGE_DIR`: SQLite storage directory, default `./data/controller`
- `PI_CONTROLLER_DRIVER_MODE`: `simulated` or `hardware`, default `simulated`
- `PI_CONTROLLER_DEFAULT_SCHEDULED_DURATION`: default scheduled feed duration in seconds
- `PI_CONTROLLER_MAX_SCHEDULED_DURATION`: max scheduled feed duration in seconds
- `PI_CONTROLLER_DEFAULT_MANUAL_DURATION`: default manual feed duration in seconds
- `PI_CONTROLLER_MAX_MANUAL_DURATION`: max manual feed duration in seconds
- `PI_CONTROLLER_SCHEDULE_TICK_SECONDS`: scheduler polling interval, default `5`
- `PI_CONTROLLER_GPIO_CHANNEL`: BCM GPIO output channel, default `17`
- `PI_CONTROLLER_GPIO_ACTIVE_HIGH`: activation polarity, default `true`

## Verification

Run controller-focused tests from the repo root:

```bash
uv run pytest tests/unit/test_pi_controller_scheduler.py
```

Run the broader Python test suite when changing shared dependencies:

```bash
uv run pytest tests/unit tests/contract tests/integration
```

## Safety

Use this service only on a trusted local aquaculture network. Hardware mode must be explicitly enabled before GPIO output is used.
