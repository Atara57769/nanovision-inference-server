import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { HealthGateway } from '@core/ports/HealthGateway'

import { useServiceHealth } from './useServiceHealth'

describe('useServiceHealth', () => {
  it('polls on mount and reflects the result', async () => {
    const healthGateway: HealthGateway = {
      checkHealth: vi.fn().mockResolvedValue({ ready: true }),
    }
    const { result } = renderHook(() => useServiceHealth(healthGateway))

    await waitFor(() => expect(result.current.health).toEqual({ ready: true }))
    expect(healthGateway.checkHealth).toHaveBeenCalledOnce()
  })

  it('markNotReady flips health to not-ready immediately, without waiting for a poll', async () => {
    const healthGateway: HealthGateway = {
      checkHealth: vi.fn().mockResolvedValue({ ready: true }),
    }
    const { result } = renderHook(() => useServiceHealth(healthGateway))
    await waitFor(() => expect(result.current.health).toEqual({ ready: true }))

    act(() => {
      result.current.markNotReady()
    })

    expect(result.current.health).toEqual({ ready: false })
  })
})
