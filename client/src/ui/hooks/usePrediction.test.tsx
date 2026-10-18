import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { PredictionResult } from '@core/entities/PredictionResult'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

import { usePrediction } from './usePrediction'

function makeFile(): File {
  return new File(['bytes'], 'cells.png', { type: 'image/png' })
}

// submit() awaits file.arrayBuffer() before it ever calls predict() — jsdom
// polyfills that via FileReader, which resolves over real macrotasks (nested
// setImmediate calls), not a microtask. A single `await Promise.resolve()`
// can't flush those; looping a few real event-loop turns can.
async function flushAsync(turns = 5): Promise<void> {
  for (let i = 0; i < turns; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

const mockResult: PredictionResult = {
  mask: { width: 2, height: 2, classGrid: new Uint8Array(4) },
  modelName: 'test-model',
  modelVersion: '1.0.0',
  inferenceTimeMs: 42,
}

describe('usePrediction', () => {
  it('does nothing when submit is called with no file selected', async () => {
    const gateway: PredictionGateway = { predict: vi.fn() }
    const { result } = renderHook(() => usePrediction(gateway))

    let returnedError: unknown
    await act(async () => {
      returnedError = await result.current.submit()
    })

    expect(returnedError).toBeNull()
    expect(gateway.predict).not.toHaveBeenCalled()
  })

  it('sets loading then result on a successful prediction', async () => {
    const gateway: PredictionGateway = {
      predict: vi.fn().mockResolvedValue(mockResult),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })

    let submitPromise: Promise<unknown>
    act(() => {
      submitPromise = result.current.submit()
    })
    expect(result.current.loading).toBe(true)

    await act(async () => {
      await submitPromise
    })

    expect(result.current.loading).toBe(false)
    expect(result.current.result).toEqual(mockResult)
    expect(result.current.lastError).toBeNull()
  })

  it('sets lastError on a rejected prediction and returns it', async () => {
    const gateway: PredictionGateway = {
      predict: vi.fn().mockRejectedValue({ kind: 'service_not_ready' }),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })

    let returnedError: unknown
    await act(async () => {
      returnedError = await result.current.submit()
    })

    expect(returnedError).toEqual({ kind: 'service_not_ready' })
    expect(result.current.lastError).toEqual({ kind: 'service_not_ready' })
    expect(result.current.loading).toBe(false)
  })

  it('clears a previous result as soon as a new submit starts, not when it resolves', async () => {
    let resolveSecondPredict: (result: PredictionResult) => void = () => {}
    const gateway: PredictionGateway = {
      predict: vi
        .fn()
        .mockResolvedValueOnce(mockResult)
        .mockImplementationOnce(
          () =>
            new Promise<PredictionResult>((resolve) => {
              resolveSecondPredict = resolve
            }),
        ),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })
    await act(async () => {
      await result.current.submit()
    })
    expect(result.current.result).toEqual(mockResult)

    act(() => {
      result.current.setSelectedFile(makeFile())
    })
    act(() => {
      void result.current.submit()
    })

    // The new file is loading — the old file's result must not still be
    // shown as if it belonged to it.
    expect(result.current.result).toBeNull()

    // Flush submit()'s file.arrayBuffer() await so predict() actually gets
    // called and rebinds resolveSecondPredict — otherwise resolving it below
    // would be a no-op against the original do-nothing closure, and this
    // test would pass without ever exercising the resolution it's named for.
    await act(flushAsync)

    const secondResult: PredictionResult = { ...mockResult, modelVersion: '2.0.0' }
    await act(async () => {
      resolveSecondPredict(secondResult)
      await flushAsync()
    })

    expect(result.current.result).toEqual(secondResult)
    expect(result.current.loading).toBe(false)
  })

  it('clears the previous result and error the moment a new file is picked, before Predict is clicked', async () => {
    const gateway: PredictionGateway = {
      predict: vi.fn().mockResolvedValue(mockResult),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })
    await act(async () => {
      await result.current.submit()
    })
    expect(result.current.result).toEqual(mockResult)

    act(() => {
      result.current.setSelectedFile(makeFile())
    })

    // No submit() yet — picking the file alone must already drop the old mask.
    expect(result.current.result).toBeNull()
    expect(result.current.lastError).toBeNull()
  })

  it('aborts a superseded request and ignores its late rejection (no stale error, no premature loading=false)', async () => {
    let rejectFirst: (error: unknown) => void = () => {}
    let firstSignal: AbortSignal | undefined
    let resolveSecond: (result: PredictionResult) => void = () => {}
    const gateway: PredictionGateway = {
      predict: vi
        .fn()
        .mockImplementationOnce((_input: unknown, signal?: AbortSignal) => {
          firstSignal = signal
          return new Promise<PredictionResult>((_resolve, reject) => {
            rejectFirst = reject
          })
        })
        .mockImplementationOnce(
          () =>
            new Promise<PredictionResult>((resolve) => {
              resolveSecond = resolve
            }),
        ),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })
    let firstSubmitPromise: Promise<unknown> = Promise.resolve(null)
    act(() => {
      firstSubmitPromise = result.current.submit()
    })
    await act(flushAsync)
    expect(gateway.predict).toHaveBeenCalledTimes(1)

    // Picking a new file mid-request must actually cancel the first
    // request, not just ignore its eventual result.
    act(() => {
      result.current.setSelectedFile(makeFile())
    })
    expect(firstSignal?.aborted).toBe(true)

    act(() => {
      void result.current.submit()
    })
    await act(flushAsync)
    expect(gateway.predict).toHaveBeenCalledTimes(2)
    expect(result.current.loading).toBe(true)

    // The stale first request rejects after being superseded.
    await act(async () => {
      rejectFirst({ kind: 'service_not_ready' })
      await flushAsync()
    })

    // Its rejection must not surface as the current error, must not clear
    // loading while the second request is still genuinely in flight, and
    // the promise it returned to its own caller must not carry the stale
    // error either.
    expect(result.current.lastError).toBeNull()
    expect(result.current.loading).toBe(true)
    await expect(firstSubmitPromise).resolves.toBeNull()

    const secondResult: PredictionResult = { ...mockResult, modelVersion: '2.0.0' }
    await act(async () => {
      resolveSecond(secondResult)
      await flushAsync()
    })

    expect(result.current.result).toEqual(secondResult)
    expect(result.current.lastError).toBeNull()
    expect(result.current.loading).toBe(false)
  })

  it('normalizes a non-PredictionError rejection to network_error', async () => {
    const gateway: PredictionGateway = {
      predict: vi.fn().mockRejectedValue(new TypeError('boom')),
    }
    const { result } = renderHook(() => usePrediction(gateway))

    act(() => {
      result.current.setSelectedFile(makeFile())
    })

    let returnedError: unknown
    await act(async () => {
      returnedError = await result.current.submit()
    })

    expect(returnedError).toEqual({ kind: 'network_error' })
  })
})
