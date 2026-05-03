# Frontend — Aquaculture Monitoring UI

React 18 + Vite + TypeScript monitoring page for live WebRTC stream viewing and detection overlay display.

## Summary (recorded 2026-05-03)

**10 tests passing** across App, hooks, and component suites.  
**Production build**: 153 kB JS + 4 kB CSS, built in ~550 ms.

## Running tests

```bash
npm test
```

## Production build

```bash
npm install
npm run build
# Output: dist/ — served by FastAPI backend
```

## Dev server

```bash
npm run dev
# Vite at http://localhost:5173
# Proxies /config, /session, /status, /ws to localhost:8000
```
