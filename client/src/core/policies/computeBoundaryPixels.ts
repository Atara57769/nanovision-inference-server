import type { SegmentationMask } from '../entities/SegmentationMask'
import { SEGMENTATION_CLASS } from '../entities/SegmentationMask'

/**
 * Returns a same-length map: the class id at each pixel on the boundary of a
 * non-background region, SEGMENTATION_CLASS.BACKGROUND (0) everywhere else.
 * Lets each region be drawn as a single-color outline over the source image
 * instead of filling every pixel. A pixel touching the image edge counts as
 * touching background.
 *
 * At an internal seam between two different non-background classes (e.g. a
 * particle's dark and bright parts), only the higher-valued class's side is
 * marked — SEGMENTATION_CLASS values are ordered so the more specific/salient
 * class (BRIGHT) wins over the more general one (DARK). Marking both sides
 * would paint the seam with one color on each side, a two-tone line instead
 * of a clean single-color outline. This way DARK gets its own outline only
 * where it touches background (the particle's outer silhouette), and BRIGHT
 * gets a complete, single-color outline all the way around — whether it
 * borders background or DARK.
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

  const outranks = (nx: number, ny: number, classId: number): boolean => classAt(nx, ny) < classId

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x
      const classId = classGrid[i]
      if (classId === SEGMENTATION_CLASS.BACKGROUND) {
        continue
      }
      const isBoundary =
        outranks(x - 1, y, classId) ||
        outranks(x + 1, y, classId) ||
        outranks(x, y - 1, classId) ||
        outranks(x, y + 1, classId)
      if (isBoundary) {
        boundary[i] = classId
      }
    }
  }

  return boundary
}
