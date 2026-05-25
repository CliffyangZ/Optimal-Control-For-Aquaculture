import type { MonitorConfig, SessionStatus } from '../types/monitoring';

export async function getConfig(): Promise<MonitorConfig> {
  const response = await fetch('/config');
  if (!response.ok) throw new Error('Failed to load monitor config');
  return response.json();
}

export async function getStatus(): Promise<SessionStatus> {
  const response = await fetch('/status');
  if (!response.ok) throw new Error('Failed to load monitor status');
  return response.json();
}

export async function startSession(): Promise<SessionStatus> {
  const response = await fetch('/session/start', { method: 'POST' });
  if (!response.ok) throw new Error('Failed to start viewer session');
  return response.json();
}

export async function heartbeat(viewerId: string): Promise<void> {
  await fetch('/session/heartbeat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ viewer_id: viewerId }),
  });
}

export async function stopSession(viewerId: string): Promise<void> {
  await fetch('/session/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ viewer_id: viewerId }),
    keepalive: true,
  });
}
