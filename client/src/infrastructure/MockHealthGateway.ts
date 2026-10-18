import type { ServiceHealth } from '@core/entities/ServiceHealth'
import type { HealthGateway } from '@core/ports/HealthGateway'

export class MockHealthGateway implements HealthGateway {
  async checkHealth(): Promise<ServiceHealth> {
    return { ready: true }
  }
}
