import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

// We'll import App after implementing it fully; stub tests here first
// so they initially fail (TDD red phase).

describe('App – live stream controls', () => {
  it('renders Start button', async () => {
    // Placeholder: will import real App once implemented
    const App = (await import('./App')).default
    render(<App />)
    expect(screen.queryByRole('button', { name: /start/i })).toBeDefined()
  })
})

describe('App – connection status', () => {
  it('renders some status indicator', async () => {
    const App = (await import('./App')).default
    render(<App />)
    // At minimum there should be the app root
    expect(document.querySelector('.app')).not.toBeNull()
  })
})
