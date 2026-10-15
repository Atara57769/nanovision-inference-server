import type { ImageInput } from '../entities/ImageInput'
import type { PredictionResult } from '../entities/PredictionResult'

export interface PredictionGateway {
  /** Rejects with a `PredictionError` from ../errors/PredictionError, never a raw Error. */
  predict(input: ImageInput): Promise<PredictionResult>
}
