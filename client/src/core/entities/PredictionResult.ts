import type { ParticleMeasurements } from './ParticleMeasurements'
import type { SegmentationMask } from './SegmentationMask'

export interface PredictionResult {
  mask: SegmentationMask
  modelName: string
  modelVersion: string
  inferenceTimeMs: number
  /** Optional: older server builds don't return this yet. */
  measurements?: ParticleMeasurements
}
