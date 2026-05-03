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
  const status = await fetchJson<SessionStatus>('/session/start', { method: 'POST' })
  // viewer_id is encoded in message field as "viewer_id:<uuid>"
  const viewer_id = status.message?.startsWith('viewer_id:')
    ? status.message.slice('viewer_id:'.length)
    : ''
  return { ...status, viewer_id }
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
