import type { SegmentationMask } from './SegmentationMask'

export interface PredictionResult {
  mask: SegmentationMask
  modelName: string
  modelVersion: string
  inferenceTimeMs: number
}
