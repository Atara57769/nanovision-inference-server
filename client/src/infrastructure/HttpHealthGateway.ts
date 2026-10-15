import type { ServiceHealth } from '@core/entities/ServiceHealth'
import type { HealthGateway } from '@core/ports/HealthGateway'

interface HealthResponseBody {
  status: string
  service_ready: boolean
}

export class HttpHealthGateway implements HealthGateway {
  private readonly baseUrl: string
  private readonly timeoutMs: number

  constructor(baseUrl: string, timeoutMs: number) {
    this.baseUrl = baseUrl
    this.timeoutMs = timeoutMs
  }

  async checkHealth(externalSignal?: AbortSignal): Promise<ServiceHealth> {
    const timeoutController = new AbortController()
    const timeoutId = setTimeout(() => timeoutController.abort(), this.timeoutMs)
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, timeoutController.signal])
      : timeoutController.signal

    try {
      const response = await fetch(`${this.baseUrl}/health`, { signal })
      if (!response.ok) {
        return { ready: false }
      }
      const body = (await response.json()) as HealthResponseBody
      return { ready: body.service_ready }
    } catch {
      return { ready: false }
    } finally {
      clearTimeout(timeoutId)
    }
  }
}
