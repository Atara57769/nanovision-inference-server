import type { HealthGateway } from '@core/ports/HealthGateway'
import type { PredictionGateway } from '@core/ports/PredictionGateway'
import { HttpHealthGateway } from '@infrastructure/HttpHealthGateway'
import { HttpPredictionGateway } from '@infrastructure/HttpPredictionGateway'
import { MockHealthGateway } from '@infrastructure/MockHealthGateway'
import { MockPredictionGateway } from '@infrastructure/MockPredictionGateway'

const DEFAULT_BASE_URL = 'http://localhost:8000'
const DEFAULT_TIMEOUT_MS = 30_000

export interface Container {
  predictionGateway: PredictionGateway
  healthGateway: HealthGateway
}

/** The only place in this app allowed to read import.meta.env. */
export function createContainer(): Container {
  const env = import.meta.env

  if (env.VITE_USE_MOCK_GATEWAYS === 'true') {
    return {
      predictionGateway: new MockPredictionGateway(),
      healthGateway: new MockHealthGateway(),
    }
  }

  const baseUrl: string = env.VITE_API_BASE_URL || DEFAULT_BASE_URL
  const timeoutMs = Number(env.VITE_API_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS

  return {
    predictionGateway: new HttpPredictionGateway(baseUrl, timeoutMs),
    healthGateway: new HttpHealthGateway(baseUrl, timeoutMs),
  }
}
