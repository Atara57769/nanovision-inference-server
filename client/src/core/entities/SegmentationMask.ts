export interface SegmentationMask {
  width: number
  height: number
  classGrid: Uint8Array
}

// Must match the server's class ids (server/app/models/classes.py): the mask
// PNG's pixel values are the model's raw class ids, used here unmapped.
export const SEGMENTATION_CLASS = {
  BACKGROUND: 0,
  DARK: 1,
  BRIGHT: 2,
} as const
