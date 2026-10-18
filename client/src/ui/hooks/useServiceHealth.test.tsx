import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ServiceHealth } from '@core/entities/ServiceHealth'
import { HEALTH_POLL_INTERVAL_MS } from '@core/policies/healthPolling'
import type { HealthGateway } from '@core/ports/HealthGateway'

import { useServiceHealth } from './useServiceHealth'

describe('useServiceHealth', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

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

  it('does not let an earlier, slower poll overwrite a later, faster one', async () => {
    let resolveFirst: (result: ServiceHealth) => void = () => {}
    let resolveSecond: (result: ServiceHealth) => void = () => {}
    const checkHealth = vi
      .fn()
      .mockImplementationOnce(
        () => new Promise<ServiceHealth>((resolve) => (resolveFirst = resolve)),
      )
      .mockImplementationOnce(
        () => new Promise<ServiceHealth>((resolve) => (resolveSecond = resolve)),
      )
    const healthGateway: HealthGateway = { checkHealth }

    vi.useFakeTimers()
    const { result } = renderHook(() => useServiceHealth(healthGateway))

    // The first poll (on mount) is still pending when the interval fires a
    // second one.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(HEALTH_POLL_INTERVAL_MS)
    })
    expect(checkHealth).toHaveBeenCalledTimes(2)

    // The newer, second poll resolves first.
    await act(async () => {
      resolveSecond({ ready: true })
      await Promise.resolve()
    })
    expect(result.current.health).toEqual({ ready: true })

    // The older, now-stale first poll resolves after it — without a
    // staleness guard this would wrongly overwrite the correct, newer value.
    await act(async () => {
      resolveFirst({ ready: false })
      await Promise.resolve()
    })
    expect(result.current.health).toEqual({ ready: true })
  })
})
