import type { PredictionResult } from '@core/entities/PredictionResult'

import { MaskCanvas } from './MaskCanvas'
import { MeasurementsPanel } from './MeasurementsPanel'

interface ResultViewProps {
  result: PredictionResult
  /** The originally uploaded image, shown under the mask outline. */
  sourceImage: File | null
}

export function ResultView({ result, sourceImage }: ResultViewProps) {
  return (
    <section className="card result" aria-label="Prediction result">
      <div className="result__header">
        <h2>Segmentation result</h2>
        <div className="result__meta">
          {/* No hardcoded "v" prefix — the server's modelVersion string owns its own format
              (e.g. "v1-step2500-78373665" vs the mock's "0.0.0-mock"); don't assume one. */}
          <span>
            {result.modelName} {result.modelVersion}
          </span>
          <span aria-hidden="true">·</span>
          <span>{result.inferenceTimeMs.toFixed(1)} ms</span>
        </div>
      </div>
      <div className="result__body">
        <MaskCanvas mask={result.mask} sourceImage={sourceImage} />
        <MeasurementsPanel measurements={result.measurements} />
      </div>
    </section>
  )
}
