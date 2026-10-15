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

  async predict(input: ImageInput): Promise<PredictionResult> {
    const formData = new FormData()
    formData.append(
      'file',
      new Blob([input.bytes], { type: input.contentType }),
      input.fileName,
    )

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs)

    let response: Response
    try {
      response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
    } catch {
      const error: PredictionError = controller.signal.aborted
        ? { kind: 'timeout' }
        : { kind: 'network_error' }
      throw error
    } finally {
      clearTimeout(timeoutId)
    }

    if (!response.ok) {
      throw await this.toPredictionError(response)
    }

    const body = (await response.json()) as PredictResponseBody

    return {
      mask: decodeMaskPng(body.mask),
      modelName: body.model_name,
      modelVersion: body.model_version,
      inferenceTimeMs: body.inference_time_ms,
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
