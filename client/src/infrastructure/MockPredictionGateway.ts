import type { ImageInput } from '@core/entities/ImageInput'
import type {
  Particle,
  ParticleMeasurements,
  StatSummary,
} from '@core/entities/ParticleMeasurements'
import type { PredictionResult } from '@core/entities/PredictionResult'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'
import type { PredictionGateway } from '@core/ports/PredictionGateway'

const MOCK_DELAY_MS = 600
const MOCK_SIZE = 4

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildMockClassGrid(): Uint8Array {
  // A simple checkerboard of background/bright/dark so the mock is visibly
  // not just a solid color when rendered.
  const classGrid = new Uint8Array(MOCK_SIZE * MOCK_SIZE)
  const classes = [
    SEGMENTATION_CLASS.BACKGROUND,
    SEGMENTATION_CLASS.BRIGHT,
    SEGMENTATION_CLASS.DARK,
  ]
  for (let i = 0; i < classGrid.length; i++) {
    classGrid[i] = classes[i % classes.length]
  }
  return classGrid
}

/** Computes a real (not hand-picked) count/mean/variance/histogram from a set
 *  of values, matching how the server's own `summarize()` behaves — so the
 *  mock's statistics are always consistent with its mock particles. `bins`
 *  defaults to 10 to match the server's own fixed bin count
 *  (server/app/analysis/statistics.py: `np.histogram(data, bins=10)`). */
function summarize(values: number[], bins = 10): StatSummary {
  const count = values.length
  if (count === 0) {
    return { count: 0, mean: null, variance: null, histogram: { binEdges: [], counts: [] } }
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / count
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + (span * i) / bins)
  const counts = new Array<number>(bins).fill(0)
  for (const value of values) {
    const binIndex = Math.min(bins - 1, Math.floor(((value - min) / span) * bins))
    counts[binIndex]++
  }
  return { count, mean, variance, histogram: { binEdges, counts } }
}

function definedOnly(values: Array<number | null>): number[] {
  return values.filter((value): value is number => value !== null)
}

// Illustrative only, shaped like the server's real /predict `measurements`
// payload (server/app/analysis/particle_analyzer.py) — not a real image's
// measurements.
const MOCK_PARTICLES: Particle[] = [
  { id: 1, centroid: [12, 9], area: 88, perimeter: 38, circularity: 0.766, cuAreaFraction: 0.62, agAreaFraction: 0.38, cuAgAreaRatio: 1.632, interfaceLength: 12, nearestParticleDistance: 15.2 },
  { id: 2, centroid: [27, 14], area: 134, perimeter: 46, circularity: 0.796, cuAreaFraction: 0.55, agAreaFraction: 0.45, cuAgAreaRatio: 1.222, interfaceLength: 16, nearestParticleDistance: 22.7 },
  { id: 3, centroid: [18, 31], area: 210, perimeter: 58, circularity: 0.785, cuAreaFraction: 0.7, agAreaFraction: 0.3, cuAgAreaRatio: 2.333, interfaceLength: 22, nearestParticleDistance: 18.4 },
  { id: 4, centroid: [41, 22], area: 156, perimeter: 50, circularity: 0.784, cuAreaFraction: 0.4, agAreaFraction: 0.6, cuAgAreaRatio: 0.667, interfaceLength: 18, nearestParticleDistance: 9.6 },
  { id: 5, centroid: [33, 44], area: 97, perimeter: 40, circularity: 0.762, cuAreaFraction: 0.58, agAreaFraction: 0.42, cuAgAreaRatio: 1.381, interfaceLength: 14, nearestParticleDistance: 30.1 },
]

const MOCK_MEASUREMENTS: ParticleMeasurements = {
  particles: MOCK_PARTICLES,
  statistics: {
    area: summarize(MOCK_PARTICLES.map((p) => p.area)),
    perimeter: summarize(MOCK_PARTICLES.map((p) => p.perimeter)),
    circularity: summarize(MOCK_PARTICLES.map((p) => p.circularity)),
    cuAreaFraction: summarize(MOCK_PARTICLES.map((p) => p.cuAreaFraction)),
    agAreaFraction: summarize(MOCK_PARTICLES.map((p) => p.agAreaFraction)),
    cuAgAreaRatio: summarize(definedOnly(MOCK_PARTICLES.map((p) => p.cuAgAreaRatio))),
    interfaceLength: summarize(MOCK_PARTICLES.map((p) => p.interfaceLength)),
    nearestParticleDistance: summarize(definedOnly(MOCK_PARTICLES.map((p) => p.nearestParticleDistance))),
  },
  // Already in the domain's short-symbol form — the real gateway normalizes
  // the server's wire vocabulary ("pixels"/"pixels_squared") to this before
  // it reaches a PredictionResult, and this mock skips that wire format
  // entirely by construction.
  units: { length: 'px', area: 'px²' },
}

export class MockPredictionGateway implements PredictionGateway {
  async predict(input: ImageInput, signal?: AbortSignal): Promise<PredictionResult> {
    void input
    void signal
    await delay(MOCK_DELAY_MS)

    return {
      mask: {
        width: MOCK_SIZE,
        height: MOCK_SIZE,
        classGrid: buildMockClassGrid(),
      },
      modelName: 'mock-model',
      modelVersion: '0.0.0-mock',
      inferenceTimeMs: MOCK_DELAY_MS,
      measurements: MOCK_MEASUREMENTS,
    }
  }
}
