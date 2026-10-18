import type { ServiceHealth } from '../entities/ServiceHealth'

export interface HealthGateway {
  /**
   * Never rejects — a network failure must resolve as `{ ready: false }`.
   * `signal`, when given, lets the caller cancel an in-flight check (e.g. on
   * effect cleanup) instead of letting a now-useless request run to completion.
   */
  checkHealth(signal?: AbortSignal): Promise<ServiceHealth>
}
