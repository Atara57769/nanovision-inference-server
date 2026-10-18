import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ImageInput } from '@core/entities/ImageInput'
import { isPredictionError } from '@core/errors/PredictionError'

import { HttpPredictionGateway } from './HttpPredictionGateway'
import { buildGrayscalePngBase64 } from './testFixtures/buildGrayscalePngBase64'

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

  // A minimal but complete statistics-body fixture — every field the real
  // server always includes, even when the test only cares about one of them.
  const emptyStatBody = {
    count: 0,
    mean: null,
    variance: null,
    histogram: { bin_edges: [], counts: [] },
  }

  it('maps a successful response, including snake_case particle measurements, to a PredictionResult', async () => {
    const mask = buildGrayscalePngBase64([0, 1, 2, 1], 2, 2)
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          mask,
          model_name: 'nanovision-seg',
          model_version: 'v1.0.0',
          inference_time_ms: 42.3,
          measurements: {
            particles: [
              {
                id: 1,
                centroid: [1.5, 0.5],
                area: 4,
                perimeter: 8,
                circularity: 0.785,
                cu_area_fraction: 0.5,
                ag_area_fraction: 0.5,
                cu_ag_area_ratio: 1,
                interface_length: 2,
                nearest_particle_distance: null,
              },
            ],
            statistics: {
              area: { count: 1, mean: 4, variance: 0, histogram: { bin_edges: [0, 8], counts: [1] } },
              perimeter: emptyStatBody,
              circularity: emptyStatBody,
              cu_area_fraction: emptyStatBody,
              ag_area_fraction: emptyStatBody,
              cu_ag_area_ratio: emptyStatBody,
              interface_length: emptyStatBody,
              nearest_particle_distance: emptyStatBody,
            },
            units: { length: 'pixels', area: 'pixels_squared' },
          },
        }),
      ),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    const result = await gateway.predict(input)

    expect(result.modelName).toBe('nanovision-seg')
    expect(result.modelVersion).toBe('v1.0.0')
    expect(result.inferenceTimeMs).toBe(42.3)
    expect(result.measurements?.particles).toEqual([
      {
        id: 1,
        centroid: [1.5, 0.5],
        area: 4,
        perimeter: 8,
        circularity: 0.785,
        cuAreaFraction: 0.5,
        agAreaFraction: 0.5,
        cuAgAreaRatio: 1,
        interfaceLength: 2,
        nearestParticleDistance: null,
      },
    ])
    expect(result.measurements?.statistics.area).toEqual({
      count: 1,
      mean: 4,
      variance: 0,
      histogram: { binEdges: [0, 8], counts: [1] },
    })
    expect(result.measurements?.units).toEqual({ length: 'pixels', area: 'pixels_squared' })
  })

  it('leaves measurements undefined when the response has no measurements field', async () => {
    const mask = buildGrayscalePngBase64([0], 1, 1)
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse(200, {
          mask,
          model_name: 'nanovision-seg',
          model_version: 'v1.0.0',
          inference_time_ms: 10,
        }),
      ),
    )
    const gateway = new HttpPredictionGateway('http://localhost', 5000)

    const result = await gateway.predict(input)

    expect(result.measurements).toBeUndefined()
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
