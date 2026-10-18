export interface Histogram {
  binEdges: number[]
  counts: number[]
}

export interface StatSummary {
  count: number
  mean: number | null
  variance: number | null
  histogram: Histogram
}

export interface Particle {
  id: number
  /** [x, y], in pixels. */
  centroid: readonly [number, number]
  area: number
  perimeter: number
  circularity: number
  cuAreaFraction: number
  agAreaFraction: number
  /** Null when the particle's Ag area is zero. */
  cuAgAreaRatio: number | null
  interfaceLength: number
  /** Null for a single-particle mask. */
  nearestParticleDistance: number | null
}

export interface MeasurementStatistics {
  area: StatSummary
  perimeter: StatSummary
  circularity: StatSummary
  cuAreaFraction: StatSummary
  agAreaFraction: StatSummary
  cuAgAreaRatio: StatSummary
  interfaceLength: StatSummary
  nearestParticleDistance: StatSummary
}

export type MeasurementField = keyof MeasurementStatistics

export interface ParticleMeasurements {
  particles: Particle[]
  statistics: MeasurementStatistics
  /** Short display symbols (e.g. "px", "px²") — the gateway normalizes the
   *  server's own unit vocabulary before it reaches the domain layer. */
  units: { length: string; area: string }
}
