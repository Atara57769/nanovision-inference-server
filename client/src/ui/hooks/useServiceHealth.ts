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

    async function poll(): Promise<void> {
      const result = await healthGateway.checkHealth(controller.signal)
      if (!cancelled) {
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
