import type { ImageInput } from '@core/entities/ImageInput'
import type {
  MeasurementStatistics,
  Particle,
  ParticleMeasurements,
  StatSummary,
} from '@core/entities/ParticleMeasurements'
import type { PredictionResult } from '@core/entities/PredictionResult'
import type { PredictionError } from '@core/errors/PredictionError'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

import { decodeMaskPng } from './pngDecode'

interface StatSummaryBody {
  count: number
  mean: number | null
  variance: number | null
  histogram: { bin_edges: number[]; counts: number[] }
}

interface ParticleBody {
  id: number
  centroid: readonly [number, number]
  area: number
  perimeter: number
  circularity: number
  cu_area_fraction: number
  ag_area_fraction: number
  cu_ag_area_ratio: number | null
  interface_length: number
  nearest_particle_distance: number | null
}

interface MeasurementsBody {
  particles: ParticleBody[]
  statistics: {
    area: StatSummaryBody
    perimeter: StatSummaryBody
    circularity: StatSummaryBody
    cu_area_fraction: StatSummaryBody
    ag_area_fraction: StatSummaryBody
    cu_ag_area_ratio: StatSummaryBody
    interface_length: StatSummaryBody
    nearest_particle_distance: StatSummaryBody
  }
  units: { length: string; area: string }
}

interface PredictResponseBody {
  mask: string
  model_name: string
  model_version: string
  inference_time_ms: number
  // Older server builds don't return this yet — optional until it's
  // everywhere.
  measurements?: MeasurementsBody
}

function mapStatSummary(body: StatSummaryBody): StatSummary {
  return {
    count: body.count,
    mean: body.mean,
    variance: body.variance,
    histogram: { binEdges: body.histogram.bin_edges, counts: body.histogram.counts },
  }
}

function mapParticle(body: ParticleBody): Particle {
  return {
    id: body.id,
    centroid: body.centroid,
    area: body.area,
    perimeter: body.perimeter,
    circularity: body.circularity,
    cuAreaFraction: body.cu_area_fraction,
    agAreaFraction: body.ag_area_fraction,
    cuAgAreaRatio: body.cu_ag_area_ratio,
    interfaceLength: body.interface_length,
    nearestParticleDistance: body.nearest_particle_distance,
  }
}

function mapStatistics(body: MeasurementsBody['statistics']): MeasurementStatistics {
  return {
    area: mapStatSummary(body.area),
    perimeter: mapStatSummary(body.perimeter),
    circularity: mapStatSummary(body.circularity),
    cuAreaFraction: mapStatSummary(body.cu_area_fraction),
    agAreaFraction: mapStatSummary(body.ag_area_fraction),
    cuAgAreaRatio: mapStatSummary(body.cu_ag_area_ratio),
    interfaceLength: mapStatSummary(body.interface_length),
    nearestParticleDistance: mapStatSummary(body.nearest_particle_distance),
  }
}

function mapMeasurements(body: MeasurementsBody | undefined): ParticleMeasurements | undefined {
  if (!body) {
    return undefined
  }
  return {
    particles: body.particles.map(mapParticle),
    statistics: mapStatistics(body.statistics),
    units: body.units,
  }
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
          measurements: mapMeasurements(body.measurements),
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
