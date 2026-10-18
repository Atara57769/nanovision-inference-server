import { useCallback, useEffect, useState } from 'react'

import type { ServiceHealth } from '@core/entities/ServiceHealth'
import { HEALTH_POLL_INTERVAL_MS } from '@core/policies/healthPolling'
import type { HealthGateway } from '@core/ports/HealthGateway'

interface UseServiceHealthResult {
  health: ServiceHealth
  /** Called by the caller when predict() itself returns service_not_ready,
   *  so the UI doesn't wait for the next poll tick to reflect it. */
  markNotReady: () => void
}

export function useServiceHealth(healthGateway: HealthGateway): UseServiceHealthResult {
  const [health, setHealth] = useState<ServiceHealth>({ ready: false })

  useEffect(() => {
    let cancelled = false
    // Aborts any poll still in flight on cleanup — without this, React
    // StrictMode's dev-only mount→cleanup→mount cycle (and any real unmount)
    // leaves an orphaned request running instead of actually cancelling it.
    const controller = new AbortController()
    // The gateway's own per-call timeout can exceed the poll interval, so two
    // polls can be in flight at once. Without this, whichever one RESOLVES
    // last would win regardless of which was SENT last, letting a slow,
    // stale poll overwrite a newer, already-applied result.
    let latestPollId = 0

    async function poll(): Promise<void> {
      const pollId = ++latestPollId
      const result = await healthGateway.checkHealth(controller.signal)
      if (!cancelled && pollId === latestPollId) {
        setHealth(result)
      }
    }

    void poll()
    const intervalId = setInterval(() => void poll(), HEALTH_POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      controller.abort()
      clearInterval(intervalId)
    }
  }, [healthGateway])

  const markNotReady = useCallback(() => {
    setHealth({ ready: false })
  }, [])

  return { health, markNotReady }
}
