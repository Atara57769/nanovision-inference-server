import type { ParticleMeasurements, StatSummary } from '@core/entities/ParticleMeasurements'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

import { CLASS_LABELS, classColorCss } from '../lib/classColors'

interface MeasurementsPanelProps {
  measurements: ParticleMeasurements | undefined
}

function formatStat(value: number | null, digits = 1): string {
  return value === null ? '—' : value.toFixed(digits)
}

/** "±" conventionally means one standard deviation, not variance (which is
 *  in squared units) — so the row table shows sqrt(variance), not variance
 *  itself. */
function formatStdDev(variance: number | null, digits = 1): string {
  return formatStat(variance === null ? null : Math.sqrt(variance), digits)
}

/** Labels each histogram bin once, reused for both the accessible summary
 *  and the visible bars. Guards against a histogram whose bin_edges aren't
 *  exactly one longer than counts (an off-by-one the server shouldn't send,
 *  but nothing here assumes it won't). */
function binLabels(histogram: StatSummary['histogram']): string[] {
  return histogram.counts.map((_, i) => {
    const lo = histogram.binEdges[i]
    const hi = histogram.binEdges[i + 1]
    return lo === undefined || hi === undefined ? '' : `${lo.toFixed(0)}–${hi.toFixed(0)}`
  })
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
  const areaBinLabels = binLabels(area.histogram)

  // Derived, not independently rounded: cuMean + agMean is always exactly 1,
  // so rounding each on its own can land the two percentages on 101 (or 99)
  // whenever the true value sits on a rounding tie. Deriving agPct from
  // cuPct keeps the pair (and the stacked bar) always summing to 100.
  const cuPct = Math.round((statistics.cuAreaFraction.mean ?? 0) * 100)
  const agPct = 100 - cuPct
  const composition = [
    { classId: SEGMENTATION_CLASS.DARK, pct: cuPct },
    { classId: SEGMENTATION_CLASS.BRIGHT, pct: agPct },
  ]

  return (
    <div className="measurements">
      <div className="measurements__count">
        {measurements.particles.length} particle{measurements.particles.length === 1 ? '' : 's'}
      </div>

      <div className="measurements__group">
        <div className="measurements__label">Area</div>
        <div className="measurements__summary-line">
          <span>
            mean {formatStat(area.mean)} {units.area}
          </span>
          <span aria-hidden="true">·</span>
          <span>std dev {formatStdDev(area.variance)}</span>
        </div>
        <div
          className="measurements__histogram"
          role="img"
          aria-label={`Area histogram: ${area.histogram.counts
            .map((count, i) => `${areaBinLabels[i]} ${units.area}: ${count}`)
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
              <span className="measurements__bin">{areaBinLabels[i]}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="measurements__group">
        <div className="measurements__label">Mean composition</div>
        <div
          className="measurements__composition"
          role="img"
          aria-label={`Mean composition: ${cuPct}% ${CLASS_LABELS[SEGMENTATION_CLASS.DARK]}, ${agPct}% ${CLASS_LABELS[SEGMENTATION_CLASS.BRIGHT]}`}
        >
          {composition.map(({ classId, pct }) => (
            <span key={classId} style={{ width: `${pct}%`, background: classColorCss(classId) }} />
          ))}
        </div>
        <div className="measurements__composition-legend">
          {composition.map(({ classId, pct }) => (
            <span key={classId}>
              <span
                className="measurements__swatch"
                style={{ background: classColorCss(classId) }}
                aria-hidden="true"
              />
              {CLASS_LABELS[classId] ?? `Class ${classId}`} {pct}%
            </span>
          ))}
        </div>
      </div>

      <dl className="measurements__table">
        {OTHER_ROWS.map(({ label, field, unit }) => {
          const stat: StatSummary = statistics[field]
          const unitSuffix = unit === 'length' ? ` ${units.length}` : ''
          return (
            <div className="measurements__row" key={field}>
              <dt>{label}</dt>
              <dd>
                {formatStat(stat.mean)}
                {unitSuffix}
                <span className="measurements__row-var"> ± {formatStdDev(stat.variance)}</span>
              </dd>
            </div>
          )
        })}
      </dl>
    </div>
  )
}
