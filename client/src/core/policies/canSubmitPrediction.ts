import type { ServiceHealth } from '../entities/ServiceHealth'

export function canSubmitPrediction(
  health: ServiceHealth,
  hasSelectedFile: boolean,
  isSubmitting: boolean,
): boolean {
  return health.ready && hasSelectedFile && !isSubmitting
}
