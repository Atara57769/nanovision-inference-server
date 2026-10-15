export interface SegmentationMask {
  width: number
  height: number
  classGrid: Uint8Array
}

export const SEGMENTATION_CLASS = {
  BACKGROUND: 0,
  BRIGHT: 1,
  DARK: 2,
} as const
