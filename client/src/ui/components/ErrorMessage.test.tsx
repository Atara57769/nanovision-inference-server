import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { PredictionError } from '@core/errors/PredictionError'

import { ErrorMessage } from './ErrorMessage'

describe('ErrorMessage', () => {
  it.each<[PredictionError, string]>([
    [{ kind: 'invalid_image', reason: 'File too large' }, 'File too large'],
    [{ kind: 'service_not_ready' }, "isn't ready"],
    [{ kind: 'prediction_failed' }, 'Prediction failed'],
    [{ kind: 'timeout' }, 'timed out'],
    [{ kind: 'network_error' }, "Can't reach the server"],
  ])('renders a message for %o', (error, expectedText) => {
    render(<ErrorMessage error={error} />)
    expect(screen.getByRole('alert')).toHaveTextContent(expectedText)
  })
})
