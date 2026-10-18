import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ImageInput } from '@core/entities/ImageInput'
import { isPredictionError } from '@core/errors/PredictionError'

import { HttpPredictionGateway } from './HttpPredictionGateway'

const input: ImageInput = {
  bytes: new Uint8Array([1, 2, 3]),
  fileName: 'test.png',
  contentType: 'image/png',
}

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('HttpPredictionGateway', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('maps a 400 response to invalid_image with the server detail as reason', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(400, { detail: 'Uploaded file is empty' })),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    await expect(gateway.predict(input)).rejects.toMatchObject({
      kind: 'invalid_image',
      reason: 'Uploaded file is empty',
    })
  })

  it('maps a 503 response to service_not_ready', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(503, {})))
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    await expect(gateway.predict(input)).rejects.toMatchObject({ kind: 'service_not_ready' })
  })

  it('maps a 500 response to prediction_failed', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(500, {})))
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    await expect(gateway.predict(input)).rejects.toMatchObject({ kind: 'prediction_failed' })
  })

  it('maps a fetch rejection to network_error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    await expect(gateway.predict(input)).rejects.toMatchObject({ kind: 'network_error' })
  })

  it('maps an aborted request to timeout, distinct from network_error', async () => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => {
              reject(new DOMException('Aborted', 'AbortError'))
            })
          }),
      ),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 1000)

    const predictPromise = gateway.predict(input)
    const assertion = expect(predictPromise).rejects.toMatchObject({ kind: 'timeout' })
    await vi.advanceTimersByTimeAsync(1000)
    await assertion
  })

  it('also times out while the response body is still being read, not just while waiting for headers', async () => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          Promise.resolve({
            ok: true,
            status: 200,
            json: () =>
              new Promise((_resolve, reject) => {
                init.signal?.addEventListener('abort', () => {
                  reject(new DOMException('Aborted', 'AbortError'))
                })
              }),
          }) as Promise<Response>,
      ),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 1000)

    const predictPromise = gateway.predict(input)
    const assertion = expect(predictPromise).rejects.toMatchObject({ kind: 'timeout' })
    await vi.advanceTimersByTimeAsync(1000)
    await assertion
  })

  it('aborts the request when the caller cancels via an external signal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => {
              reject(new DOMException('Aborted', 'AbortError'))
            })
          }),
      ),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 30_000)
    const externalController = new AbortController()

    const predictPromise = gateway.predict(input, externalController.signal)
    externalController.abort()

    await expect(predictPromise).rejects.toMatchObject({ kind: 'network_error' })
  })

  it('rejects with a value isPredictionError recognizes', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(500, {})))
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    await gateway.predict(input).catch((error: unknown) => {
      expect(isPredictionError(error)).toBe(true)
    })
  })
})
