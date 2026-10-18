import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

type Rgb = readonly [number, number, number]

export const CLASS_COLORS: Record<number, Rgb> = {
  [SEGMENTATION_CLASS.BRIGHT]: [250, 204, 21],
  [SEGMENTATION_CLASS.DARK]: [99, 61, 191],
}

export const FALLBACK_CLASS_COLOR: Rgb = [220, 38, 38]

// Per server/README.md: dark pixels represent Cu, bright pixels represent Ag.
export const CLASS_LABELS: Record<number, string> = {
  [SEGMENTATION_CLASS.DARK]: 'Cu',
  [SEGMENTATION_CLASS.BRIGHT]: 'Ag',
}

export function classColorCss(classId: number): string {
  const [r, g, b] = CLASS_COLORS[classId] ?? FALLBACK_CLASS_COLOR
  return `rgb(${r}, ${g}, ${b})`
}
