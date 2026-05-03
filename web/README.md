# Web Module

FastAPI backend and React/Vite frontend for low-latency aquaculture monitoring.

## Normal operator flow

```bash
# 1. Start MediaMTX
docker compose -f compose.mediamtx.desktop.yml up

# 2. Build the React frontend
cd web/frontend
npm install
npm run build
cd ../..

# 3. Start the FastAPI backend
uvicorn web.app:app --host 0.0.0.0 --port 8000

# 4. Open http://localhost:8000 in Chrome or Edge
```

## Frontend development flow

```bash
# Start MediaMTX and the backend first, then:
cd web/frontend
npm install
npm run dev   # Vite dev server at http://localhost:5173
```

API requests (vite.config.ts proxy) forward to the FastAPI backend at localhost:8000.

## Running tests

**Backend**:
```bash
uv run python -m pytest tests/contract/ tests/integration/ tests/unit/
```

**Frontend**:
```bash
cd web/frontend
npm test
```

## Security boundary

Local trusted-network only. Do not expose FastAPI, MediaMTX WebRTC, or overlay WebSockets to public networks. v1 does not include login, RBAC, recording, or persistent overlay storage.
