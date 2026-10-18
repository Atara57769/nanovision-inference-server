import { afterEach, describe, expect, it, vi } from 'vitest'

import { HttpHealthGateway } from './HttpHealthGateway'

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('HttpHealthGateway', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('resolves ready:true when the server reports service_ready:true', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, { status: 'ok', service_ready: true })),
    )
    const gateway = new HttpHealthGateway('http://localhost', 5000)

    await expect(gateway.checkHealth()).resolves.toEqual({ ready: true })
  })

  it('resolves ready:false when the server reports service_ready:false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(200, { status: 'ok', service_ready: false })),
    )
    const gateway = new HttpHealthGateway('http://localhost', 5000)

    await expect(gateway.checkHealth()).resolves.toEqual({ ready: false })
  })

  it('resolves ready:false instead of rejecting when the network call fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')))
    const gateway = new HttpHealthGateway('http://localhost', 5000)

    await expect(gateway.checkHealth()).resolves.toEqual({ ready: false })
  })

  it('aborts the underlying fetch when the caller cancels via its own signal', async () => {
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
    const gateway = new HttpHealthGateway('http://localhost', 5000)
    const controller = new AbortController()

    const checkPromise = gateway.checkHealth(controller.signal)
    controller.abort()

    await expect(checkPromise).resolves.toEqual({ ready: false })
  })
})
