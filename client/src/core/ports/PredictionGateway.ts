import type { ImageInput } from '../entities/ImageInput'
import type { PredictionResult } from '../entities/PredictionResult'

export interface PredictionGateway {
  /**
   * Rejects with a `PredictionError` from ../errors/PredictionError, never a raw Error.
   * `signal`, when given, lets the caller cancel an in-flight prediction (e.g. because a
   * newer request superseded it) instead of letting a now-useless request — and the
   * backend inference behind it — run to completion.
   */
  predict(input: ImageInput, signal?: AbortSignal): Promise<PredictionResult>
}
