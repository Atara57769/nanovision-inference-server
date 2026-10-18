import { describe, expect, it } from 'vitest'

import type { ServiceHealth } from '../entities/ServiceHealth'
import { canSubmitPrediction } from './canSubmitPrediction'

describe('canSubmitPrediction', () => {
  const ready: ServiceHealth = { ready: true }
  const notReady: ServiceHealth = { ready: false }

  it('allows submission when ready, a file is selected, and nothing is in flight', () => {
    expect(canSubmitPrediction(ready, true, false)).toBe(true)
  })

  it('blocks submission when the service is not ready', () => {
    expect(canSubmitPrediction(notReady, true, false)).toBe(false)
  })

  it('blocks submission when no file is selected', () => {
    expect(canSubmitPrediction(ready, false, false)).toBe(false)
  })

  it('blocks submission when a prediction is already in flight', () => {
    expect(canSubmitPrediction(ready, true, true)).toBe(false)
  })

  it('blocks submission when nothing holds', () => {
    expect(canSubmitPrediction(notReady, false, true)).toBe(false)
  })
})
