import type { ParticleMeasurements, StatSummary } from '@core/entities/ParticleMeasurements'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

import { classColorCss } from '../lib/classColors'

interface MeasurementsPanelProps {
  measurements: ParticleMeasurements | undefined
}

const SHORT_UNIT: Record<string, string> = {
  pixels: 'px',
  pixels_squared: 'px²',
}

function shortUnit(unit: string): string {
  return SHORT_UNIT[unit] ?? unit
}

function formatStat(value: number | null, digits = 1): string {
  return value === null ? '—' : value.toFixed(digits)
}

const OTHER_ROWS: ReadonlyArray<{
  label: string
  field: 'perimeter' | 'circularity' | 'interfaceLength' | 'cuAgAreaRatio' | 'nearestParticleDistance'
  unit?: 'length'
}> = [
  { label: 'Perimeter', field: 'perimeter', unit: 'length' },
  { label: 'Circularity', field: 'circularity' },
  { label: 'Interface length', field: 'interfaceLength', unit: 'length' },
  { label: 'Cu/Ag area ratio', field: 'cuAgAreaRatio' },
  { label: 'Nearest particle', field: 'nearestParticleDistance', unit: 'length' },
]

export function MeasurementsPanel({ measurements }: MeasurementsPanelProps) {
  if (!measurements || measurements.particles.length === 0) {
    return null
  }

  const { statistics, units } = measurements
  const area = statistics.area
  const areaMaxCount = Math.max(...area.histogram.counts, 1)
  const areaUnit = shortUnit(units.area)
  const lengthUnit = shortUnit(units.length)

  const cuMean = statistics.cuAreaFraction.mean ?? 0
  const agMean = statistics.agAreaFraction.mean ?? 0
  const cuPct = Math.round(cuMean * 100)
  const agPct = Math.round(agMean * 100)

  return (
    <div className="measurements">
      <div className="measurements__count">
        {measurements.particles.length} particle{measurements.particles.length === 1 ? '' : 's'}
      </div>

      <div className="measurements__group">
        <div className="measurements__label">Area</div>
        <div className="measurements__summary-line">
          <span>
            mean {formatStat(area.mean)} {areaUnit}
          </span>
          <span aria-hidden="true">·</span>
          <span>variance {formatStat(area.variance)}</span>
        </div>
        <div
          className="measurements__histogram"
          role="img"
          aria-label={`Area histogram: ${area.histogram.counts
            .map((count, i) => `${area.histogram.binEdges[i].toFixed(0)}-${area.histogram.binEdges[i + 1].toFixed(0)} ${areaUnit}: ${count}`)
            .join(', ')}`}
        >
          {area.histogram.counts.map((count, i) => (
            <div className="measurements__bar-col" key={i}>
              <div className="measurements__bar-track">
                <div
                  className="measurements__bar"
                  style={{
                    height: `${(count / areaMaxCount) * 100}%`,
                    background: 'var(--accent)',
                  }}
                />
              </div>
              <span className="measurements__bin">
                {area.histogram.binEdges[i].toFixed(0)}–{area.histogram.binEdges[i + 1].toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="measurements__group">
        <div className="measurements__label">Mean composition</div>
        <div
          className="measurements__composition"
          role="img"
          aria-label={`Mean composition: ${cuPct}% Cu, ${agPct}% Ag`}
        >
          <span style={{ width: `${cuPct}%`, background: classColorCss(SEGMENTATION_CLASS.DARK) }} />
          <span style={{ width: `${agPct}%`, background: classColorCss(SEGMENTATION_CLASS.BRIGHT) }} />
        </div>
        <div className="measurements__composition-legend">
          <span>
            <span
              className="measurements__swatch"
              style={{ background: classColorCss(SEGMENTATION_CLASS.DARK) }}
              aria-hidden="true"
            />
            Cu {cuPct}%
          </span>
          <span>
            <span
              className="measurements__swatch"
              style={{ background: classColorCss(SEGMENTATION_CLASS.BRIGHT) }}
              aria-hidden="true"
            />
            Ag {agPct}%
          </span>
        </div>
      </div>

      <dl className="measurements__table">
        {OTHER_ROWS.map(({ label, field, unit }) => {
          const stat: StatSummary = statistics[field]
          const unitSuffix = unit === 'length' ? ` ${lengthUnit}` : ''
          return (
            <div className="measurements__row" key={field}>
              <dt>{label}</dt>
              <dd>
                {formatStat(stat.mean)}
                {unitSuffix}
                <span className="measurements__row-var"> ± {formatStat(stat.variance)}</span>
              </dd>
            </div>
          )
        })}
      </dl>
    </div>
  )
}
