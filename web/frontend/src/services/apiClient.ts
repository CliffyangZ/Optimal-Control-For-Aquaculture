import type { RuntimeConfig, SessionStatus } from '../types/streaming'

const BASE = ''

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${res.statusText} for ${path}`)
  }
  return res.json() as Promise<T>
}

export async function getConfig(): Promise<RuntimeConfig> {
  return fetchJson<RuntimeConfig>('/config')
}

export async function startSession(): Promise<SessionStatus & { viewer_id: string }> {
  const raw = await fetchJson<SessionStatus & { viewer_id?: string }>('/session/start', { method: 'POST' })
  const viewer_id = raw.viewer_id ?? ''
  return { ...raw, viewer_id }
}

export async function stopSession(viewer_id?: string): Promise<SessionStatus> {
  return fetchJson<SessionStatus>('/session/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ viewer_id: viewer_id ?? null }),
  })
}

export async function getStatus(): Promise<SessionStatus> {
  return fetchJson<SessionStatus>('/status')
}
