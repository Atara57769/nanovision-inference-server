import type { ImageInput } from '@core/entities/ImageInput'
import type { PredictionResult } from '@core/entities/PredictionResult'
import type { PredictionError } from '@core/errors/PredictionError'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

import { decodeMaskPng } from './pngDecode'

interface PredictResponseBody {
  mask: string
  model_name: string
  model_version: string
  inference_time_ms: number
}

export class HttpPredictionGateway implements PredictionGateway {
  private readonly baseUrl: string
  private readonly timeoutMs: number

  constructor(baseUrl: string, timeoutMs: number) {
    this.baseUrl = baseUrl
    this.timeoutMs = timeoutMs
  }

  async predict(input: ImageInput, externalSignal?: AbortSignal): Promise<PredictionResult> {
    const formData = new FormData()
    formData.append(
      'file',
      new Blob([input.bytes], { type: input.contentType }),
      input.fileName,
    )

    const timeoutController = new AbortController()
    // Covers the whole request, not just fetch() itself — cleared only once
    // the body has been fully read (or the request has failed outright), so
    // a slow-trickling response body is bounded by timeoutMs too, not just
    // the time to receive headers.
    const timeoutId = setTimeout(() => timeoutController.abort(), this.timeoutMs)
    const signal = externalSignal
      ? AbortSignal.any([externalSignal, timeoutController.signal])
      : timeoutController.signal

    try {
      let response: Response
      try {
        response = await fetch(`${this.baseUrl}/predict`, {
          method: 'POST',
          body: formData,
          signal,
        })
      } catch {
        const error: PredictionError = timeoutController.signal.aborted
          ? { kind: 'timeout' }
          : { kind: 'network_error' }
        throw error
      }

      if (!response.ok) {
        throw await this.toPredictionError(response)
      }

      try {
        const body = (await response.json()) as PredictResponseBody

        return {
          mask: decodeMaskPng(body.mask),
          modelName: body.model_name,
          modelVersion: body.model_version,
          inferenceTimeMs: body.inference_time_ms,
        }
      } catch {
        // Aborting mid-body-read rejects response.json() too — report that
        // as a timeout, same as aborting before fetch() even resolves.
        if (timeoutController.signal.aborted) {
          const error: PredictionError = { kind: 'timeout' }
          throw error
        }
        // A malformed JSON body or an undecodable mask on an otherwise-ok
        // response is a bad response, not a network/timeout failure — but it
        // must still reject with a PredictionError (never a raw Error), per
        // this class's contract in ../core/ports/PredictionGateway.ts.
        const error: PredictionError = { kind: 'prediction_failed' }
        throw error
      }
    } finally {
      clearTimeout(timeoutId)
    }
  }

  private async toPredictionError(response: Response): Promise<PredictionError> {
    if (response.status === 400) {
      const body = (await response.json().catch(() => null)) as { detail?: string } | null
      return { kind: 'invalid_image', reason: body?.detail ?? 'Invalid image' }
    }
    if (response.status === 503) {
      return { kind: 'service_not_ready' }
    }
    return { kind: 'prediction_failed' }
  }
}
