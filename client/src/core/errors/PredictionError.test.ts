import { describe, expect, it } from 'vitest'

import { isPredictionError } from './PredictionError'

describe('isPredictionError', () => {
  it('accepts every known error kind', () => {
    expect(isPredictionError({ kind: 'invalid_image', reason: 'too big' })).toBe(true)
    expect(isPredictionError({ kind: 'service_not_ready' })).toBe(true)
    expect(isPredictionError({ kind: 'prediction_failed' })).toBe(true)
    expect(isPredictionError({ kind: 'timeout' })).toBe(true)
    expect(isPredictionError({ kind: 'network_error' })).toBe(true)
  })

  it('rejects a raw Error instance', () => {
    expect(isPredictionError(new Error('boom'))).toBe(false)
  })

  it('rejects an object with an unknown kind', () => {
    expect(isPredictionError({ kind: 'something_else' })).toBe(false)
  })

  it('rejects non-objects', () => {
    expect(isPredictionError(null)).toBe(false)
    expect(isPredictionError(undefined)).toBe(false)
    expect(isPredictionError('network_error')).toBe(false)
  })
})
