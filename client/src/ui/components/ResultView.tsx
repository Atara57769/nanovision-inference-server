import type { PredictionResult } from '@core/entities/PredictionResult'

import { MaskCanvas } from './MaskCanvas'

interface ResultViewProps {
  result: PredictionResult
  /** The originally uploaded image, shown under the mask outline. */
  sourceImage: File | null
}

export function ResultView({ result, sourceImage }: ResultViewProps) {
  return (
    <section className="card result" aria-label="Prediction result">
      <dl>
        <div className="result__stat">
          <dt>Model</dt>
          <dd>
            {/* No hardcoded "v" prefix — the server's modelVersion string owns its own format
                (e.g. "v1-step2500-78373665" vs the mock's "0.0.0-mock"); don't assume one. */}
            {result.modelName} {result.modelVersion}
          </dd>
        </div>
        <div className="result__stat">
          <dt>Inference time</dt>
          <dd>{result.inferenceTimeMs.toFixed(1)} ms</dd>
        </div>
      </dl>
      <MaskCanvas mask={result.mask} sourceImage={sourceImage} />
    </section>
  )
}
