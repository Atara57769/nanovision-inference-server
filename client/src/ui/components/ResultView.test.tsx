import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { PredictionResult } from '@core/entities/PredictionResult'

import { ResultView } from './ResultView'

describe('ResultView', () => {
  it('renders model info verbatim (no assumed "v" prefix) and a mask canvas sized to the mask', () => {
    const result: PredictionResult = {
      mask: { width: 8, height: 4, classGrid: new Uint8Array(32) },
      modelName: 'nanovision-seg',
      modelVersion: 'v1-step2500-78373665',
      inferenceTimeMs: 123.456,
    }

    render(<ResultView result={result} sourceImage={null} />)

    expect(screen.getByText('nanovision-seg v1-step2500-78373665')).toBeInTheDocument()
    expect(screen.getByText('123.5 ms')).toBeInTheDocument()

    const canvas = screen.getByRole('img', { name: 'Segmentation mask' })
    expect(canvas).toHaveAttribute('width', '8')
    expect(canvas).toHaveAttribute('height', '4')
  })
})
