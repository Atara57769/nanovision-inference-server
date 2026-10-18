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
        <dl className="result__meta">
          <div>
            <dt className="sr-only">Model</dt>
            {/* No hardcoded "v" prefix — the server's modelVersion string owns its own format
                (e.g. "v1-step2500-78373665" vs the mock's "0.0.0-mock"); don't assume one. */}
            <dd>
              {result.modelName} {result.modelVersion}
            </dd>
          </div>
          <span aria-hidden="true">·</span>
          <div>
            <dt className="sr-only">Inference time</dt>
            <dd>{result.inferenceTimeMs.toFixed(1)} ms</dd>
          </div>
        </dl>
      </div>
      <div className="result__body">
        <MaskCanvas mask={result.mask} sourceImage={sourceImage} />
        <MeasurementsPanel measurements={result.measurements} />
      </div>
    </section>
  )
}
