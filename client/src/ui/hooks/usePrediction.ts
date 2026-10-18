import { useCallback, useRef, useState } from 'react'

import type { ImageInput } from '@core/entities/ImageInput'
import type { PredictionResult } from '@core/entities/PredictionResult'
import { isPredictionError, type PredictionError } from '@core/errors/PredictionError'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

interface UsePredictionResult {
  selectedFile: File | null
  setSelectedFile: (file: File | null) => void
  loading: boolean
  lastError: PredictionError | null
  result: PredictionResult | null
  /** Returns the error for the caller to react to (e.g. service_not_ready
   *  should update ServiceHealth immediately), or null on success. */
  submit: () => Promise<PredictionError | null>
}

async function toImageInput(file: File): Promise<ImageInput> {
  const bytes = new Uint8Array(await file.arrayBuffer())
  return { bytes, fileName: file.name, contentType: file.type }
}

export function usePrediction(predictionGateway: PredictionGateway): UsePredictionResult {
  const [selectedFile, setSelectedFileState] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [lastError, setLastError] = useState<PredictionError | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  // Bumped on every setSelectedFile/submit so a submit() in flight can tell,
  // once it resolves, whether it's still the one the user cares about — the
  // file input isn't disabled fast enough to rule out selecting a new file
  // mid-request, and a stale response must not overwrite the newer file's
  // (empty) result/error/loading state with the old file's, nor hand the
  // caller a stale error to react to.
  const requestIdRef = useRef(0)
  // The in-flight request's own cancellation, so a superseded request is
  // actually aborted (freeing the backend from running inference on an
  // image the user already abandoned) instead of just having its eventual
  // result ignored.
  const abortControllerRef = useRef<AbortController | null>(null)

  // Clearing result/lastError here too (not just at submit()) — otherwise
  // picking a new file still shows it paired with the previous file's mask
  // or error until Predict is clicked.
  const setSelectedFile = useCallback((file: File | null) => {
    requestIdRef.current += 1
    abortControllerRef.current?.abort()
    setSelectedFileState(file)
    setResult(null)
    setLastError(null)
  }, [])

  const submit = useCallback(async (): Promise<PredictionError | null> => {
    if (!selectedFile) {
      return null
    }

    const requestId = ++requestIdRef.current
    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller
    const isCurrent = () => requestIdRef.current === requestId

    setLoading(true)
    setLastError(null)
    setResult(null)

    try {
      const input = await toImageInput(selectedFile)
      const predictionResult = await predictionGateway.predict(input, controller.signal)
      if (!isCurrent()) {
        return null
      }
      setResult(predictionResult)
      return null
    } catch (error) {
      if (!isCurrent()) {
        return null
      }
      const predictionError: PredictionError = isPredictionError(error)
        ? error
        : { kind: 'network_error' }
      setLastError(predictionError)
      return predictionError
    } finally {
      if (isCurrent()) {
        setLoading(false)
      }
    }
  }, [selectedFile, predictionGateway])

  return { selectedFile, setSelectedFile, loading, lastError, result, submit }
}
