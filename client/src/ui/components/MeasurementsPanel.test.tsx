import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { ParticleMeasurements, StatSummary } from '@core/entities/ParticleMeasurements'

import { MeasurementsPanel } from './MeasurementsPanel'

const emptyStat: StatSummary = {
  count: 0,
  mean: null,
  variance: null,
  histogram: { binEdges: [], counts: [] },
}

const measurements: ParticleMeasurements = {
  particles: [
    {
      id: 1,
      centroid: [1, 1],
      area: 100,
      perimeter: 40,
      circularity: 0.78,
      cuAreaFraction: 0.6,
      agAreaFraction: 0.4,
      cuAgAreaRatio: 1.5,
      interfaceLength: 10,
      nearestParticleDistance: null,
    },
  ],
  statistics: {
    area: { count: 1, mean: 100, variance: 0, histogram: { binEdges: [80, 120], counts: [1] } },
    perimeter: { count: 1, mean: 40, variance: 0, histogram: { binEdges: [], counts: [] } },
    circularity: emptyStat,
    cuAreaFraction: { count: 1, mean: 0.6, variance: 0, histogram: { binEdges: [], counts: [] } },
    agAreaFraction: { count: 1, mean: 0.4, variance: 0, histogram: { binEdges: [], counts: [] } },
    cuAgAreaRatio: { count: 1, mean: 1.5, variance: 0, histogram: { binEdges: [], counts: [] } },
    interfaceLength: emptyStat,
    nearestParticleDistance: emptyStat,
  },
  // Already-mapped short symbols — HttpPredictionGateway owns converting the
  // server's raw "pixels"/"pixels_squared" wire vocabulary to these, so the
  // panel itself never deals with that conversion (see its own test suite
  // for that mapping).
  units: { length: 'px', area: 'px²' },
}

describe('MeasurementsPanel', () => {
  it('renders nothing when there are no measurements yet', () => {
    const { container } = render(<MeasurementsPanel measurements={undefined} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders nothing when the mask has no particles', () => {
    const { container } = render(
      <MeasurementsPanel measurements={{ ...measurements, particles: [] }} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('renders particle count, area stats (in short px² units), and a histogram', () => {
    render(<MeasurementsPanel measurements={measurements} />)

    expect(screen.getByText('1 particle')).toBeInTheDocument()
    expect(screen.getByText('mean 100.0 px²')).toBeInTheDocument()
    expect(screen.getByText('std dev 0.0')).toBeInTheDocument()
    expect(
      screen.getByRole('img', { name: 'Area histogram: 80–120 px²: 1' }),
    ).toBeInTheDocument()
  })

  it('renders the mean Cu/Ag composition from the fraction statistics', () => {
    render(<MeasurementsPanel measurements={measurements} />)

    expect(
      screen.getByRole('img', { name: 'Mean composition: 60% Cu, 40% Ag' }),
    ).toBeInTheDocument()
    expect(screen.getByText('Cu 60%')).toBeInTheDocument()
    expect(screen.getByText('Ag 40%')).toBeInTheDocument()
  })

  it('shows a dash for null stats, e.g. nearest-particle distance on a single particle', () => {
    render(<MeasurementsPanel measurements={measurements} />)

    const row = screen.getByText('Nearest particle').closest('.measurements__row')
    expect(row).toHaveTextContent('— px ± —')
  })

  it('derives the composition percentages so they always sum to 100', () => {
    const tiedMeasurements: ParticleMeasurements = {
      ...measurements,
      statistics: {
        ...measurements.statistics,
        cuAreaFraction: { ...measurements.statistics.cuAreaFraction, mean: 0.505 },
        agAreaFraction: { ...measurements.statistics.agAreaFraction, mean: 0.495 },
      },
    }
    render(<MeasurementsPanel measurements={tiedMeasurements} />)

    expect(
      screen.getByRole('img', { name: 'Mean composition: 51% Cu, 49% Ag' }),
    ).toBeInTheDocument()
  })
})
