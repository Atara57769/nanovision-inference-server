import type { PredictionError } from '@core/errors/PredictionError'

interface ErrorMessageProps {
  error: PredictionError
}

export function ErrorMessage({ error }: ErrorMessageProps) {
  switch (error.kind) {
    case 'invalid_image':
      return (
        <p className="alert" role="alert">
          Invalid image: {error.reason}
        </p>
      )
    case 'service_not_ready':
      return (
        <p className="alert" role="alert">
          The prediction service isn't ready yet. Try again shortly.
        </p>
      )
    case 'prediction_failed':
      return (
        <p className="alert" role="alert">
          Prediction failed. Please try again.
        </p>
      )
    case 'timeout':
      return (
        <p className="alert" role="alert">
          The request timed out. Please try again.
        </p>
      )
    case 'network_error':
      return (
        <p className="alert" role="alert">
          Can't reach the server. Check your connection.
        </p>
      )
    default: {
      // Exhaustiveness check: fails to compile if a PredictionError kind is
      // ever added here without a matching case above.
      const exhaustiveCheck: never = error
      return exhaustiveCheck
    }
  }
}
