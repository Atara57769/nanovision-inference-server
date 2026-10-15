import type { ImageInput } from '@core/entities/ImageInput'
import type { PredictionResult } from '@core/entities/PredictionResult'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

const MOCK_DELAY_MS = 600
const MOCK_SIZE = 4

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildMockClassGrid(): Uint8Array {
  // A simple checkerboard of background/bright/dark so the mock is visibly
  // not just a solid color when rendered.
  const classGrid = new Uint8Array(MOCK_SIZE * MOCK_SIZE)
  const classes = [
    SEGMENTATION_CLASS.BACKGROUND,
    SEGMENTATION_CLASS.BRIGHT,
    SEGMENTATION_CLASS.DARK,
  ]
  for (let i = 0; i < classGrid.length; i++) {
    classGrid[i] = classes[i % classes.length]
  }
  return classGrid
}

export class MockPredictionGateway implements PredictionGateway {
  async predict(input: ImageInput): Promise<PredictionResult> {
    void input
    await delay(MOCK_DELAY_MS)

    return {
      mask: {
        width: MOCK_SIZE,
        height: MOCK_SIZE,
        classGrid: buildMockClassGrid(),
      },
      modelName: 'mock-model',
      modelVersion: '0.0.0-mock',
      inferenceTimeMs: MOCK_DELAY_MS,
    }
  }
}
