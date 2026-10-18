import { canSubmitPrediction } from '@core/policies/canSubmitPrediction'
import type { HealthGateway } from '@core/ports/HealthGateway'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

import { ErrorMessage } from './components/ErrorMessage'
import { ResultView } from './components/ResultView'
import { UploadForm } from './components/UploadForm'
import { usePrediction } from './hooks/usePrediction'
import { useServiceHealth } from './hooks/useServiceHealth'

interface AppProps {
  predictionGateway: PredictionGateway
  healthGateway: HealthGateway
}

export function App({ predictionGateway, healthGateway }: AppProps) {
  const { health, markNotReady } = useServiceHealth(healthGateway)
  const { selectedFile, setSelectedFile, loading, lastError, result, submit } =
    usePrediction(predictionGateway)

  const canSubmit = canSubmitPrediction(health, selectedFile !== null, loading)

  async function handleSubmit() {
    const error = await submit()
    // Reflect service_not_ready immediately instead of waiting for the next
    // health poll tick — the decision made while planning this layer.
    if (error?.kind === 'service_not_ready') {
      markNotReady()
    }
  }

  return (
    <main className="page">
      <header className="page__header">
        <h1>NanoVision Inference</h1>
        <p className="page__subtitle">Upload a SEM image to segment Cu–Ag particles</p>
      </header>
      <section className="card">
        <UploadForm
          selectedFile={selectedFile}
          onSelectFile={setSelectedFile}
          onSubmit={handleSubmit}
          canSubmit={canSubmit}
          loading={loading}
        />
        {lastError && <ErrorMessage error={lastError} />}
      </section>
      {result && <ResultView result={result} sourceImage={selectedFile} />}
    </main>
  )
}
