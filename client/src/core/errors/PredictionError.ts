export type PredictionError =
  | { kind: 'invalid_image'; reason: string }
  | { kind: 'service_not_ready' }
  | { kind: 'prediction_failed' }
  | { kind: 'timeout' }
  | { kind: 'network_error' }

const PREDICTION_ERROR_KINDS: ReadonlySet<PredictionError['kind']> = new Set([
  'invalid_image',
  'service_not_ready',
  'prediction_failed',
  'timeout',
  'network_error',
])

/** Narrows a `catch` block's `unknown` at the infrastructure→ui trust boundary. */
export function isPredictionError(value: unknown): value is PredictionError {
  return (
    typeof value === 'object' &&
    value !== null &&
    'kind' in value &&
    typeof value.kind === 'string' &&
    PREDICTION_ERROR_KINDS.has(value.kind as PredictionError['kind'])
  )
}
