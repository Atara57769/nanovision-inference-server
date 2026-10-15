import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { PredictionResult } from '@core/entities/PredictionResult'
import type { HealthGateway } from '@core/ports/HealthGateway'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

import { App } from './App'

const mockResult: PredictionResult = {
  mask: { width: 1, height: 1, classGrid: new Uint8Array([1]) },
  modelName: 'nanovision-seg',
  modelVersion: 'v1.0.0',
  inferenceTimeMs: 10,
}

function readyHealthGateway(): HealthGateway {
  return { checkHealth: vi.fn().mockResolvedValue({ ready: true }) }
}

describe('App', () => {
  it('disables submit until ready and a file is selected, then runs a full predict cycle', async () => {
    const user = userEvent.setup()
    const predictionGateway: PredictionGateway = {
      predict: vi.fn().mockResolvedValue(mockResult),
    }
    const healthGateway = readyHealthGateway()

    render(<App predictionGateway={predictionGateway} healthGateway={healthGateway} />)

    const button = screen.getByRole('button', { name: 'Predict' })
    expect(button).toBeDisabled()

    const file = new File(['bytes'], 'cells.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Select image'), file)
    await waitFor(() => expect(button).toBeEnabled())

    await user.click(button)

    expect(await screen.findByText('nanovision-seg v1.0.0')).toBeInTheDocument()
    expect(predictionGateway.predict).toHaveBeenCalledOnce()
  })

  it('reflects service_not_ready immediately after a failed submit, without waiting for the next poll', async () => {
    const user = userEvent.setup()
    const predictionGateway: PredictionGateway = {
      predict: vi.fn().mockRejectedValue({ kind: 'service_not_ready' }),
    }
    const healthGateway = readyHealthGateway()

    render(<App predictionGateway={predictionGateway} healthGateway={healthGateway} />)

    const file = new File(['bytes'], 'cells.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Select image'), file)

    const button = screen.getByRole('button', { name: 'Predict' })
    await waitFor(() => expect(button).toBeEnabled())
    await user.click(button)

    expect(await screen.findByRole('alert')).toHaveTextContent("isn't ready")
    // The button is disabled again although the gateway was never polled a
    // second time — proof health came from the submit() result, not a poll.
    expect(button).toBeDisabled()
    expect(healthGateway.checkHealth).toHaveBeenCalledOnce()
  })
})
