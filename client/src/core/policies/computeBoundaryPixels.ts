import type { SegmentationMask } from '../entities/SegmentationMask'
import { SEGMENTATION_CLASS } from '../entities/SegmentationMask'

/**
 * Returns a same-length map: the class id at each pixel on the boundary of a
 * non-background region, SEGMENTATION_CLASS.BACKGROUND (0) everywhere else.
 * Lets the mask be drawn as a thin outline over the source image instead of
 * filling every pixel. A pixel touching the image edge counts as a boundary,
 * same as one touching background.
 */
export function computeBoundaryPixels(mask: SegmentationMask): Uint8Array {
  const { width, height, classGrid } = mask
  const boundary = new Uint8Array(classGrid.length)

  const classAt = (x: number, y: number): number => {
    if (x < 0 || x >= width || y < 0 || y >= height) {
      return SEGMENTATION_CLASS.BACKGROUND
    }
    return classGrid[y * width + x]
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const classId = classGrid[i]
      if (classId === SEGMENTATION_CLASS.BACKGROUND) {
        continue
      }
      const isBoundary =
        classId !== classAt(x - 1, y) ||
        classId !== classAt(x + 1, y) ||
        classId !== classAt(x, y - 1) ||
        classId !== classAt(x, y + 1)
      if (isBoundary) {
        boundary[i] = classId
      }
    }
  }

  return boundary
}
