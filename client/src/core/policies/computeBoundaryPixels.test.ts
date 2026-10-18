import { describe, expect, it } from 'vitest'

import type { SegmentationMask } from '../entities/SegmentationMask'
import { SEGMENTATION_CLASS } from '../entities/SegmentationMask'
import { computeBoundaryPixels } from './computeBoundaryPixels'

function mask(width: number, height: number, values: number[]): SegmentationMask {
  return { width, height, classGrid: Uint8Array.from(values) }
}

const { BACKGROUND, BRIGHT, DARK } = SEGMENTATION_CLASS

describe('computeBoundaryPixels', () => {
  it('marks nothing when the whole mask is background', () => {
    const m = mask(3, 3, [
      BACKGROUND, BACKGROUND, BACKGROUND,
      BACKGROUND, BACKGROUND, BACKGROUND,
      BACKGROUND, BACKGROUND, BACKGROUND,
    ])

    expect(Array.from(computeBoundaryPixels(m))).toEqual(new Array(9).fill(0))
  })

  it('marks a single non-background pixel surrounded by background', () => {
    const m = mask(3, 3, [
      BACKGROUND, BACKGROUND, BACKGROUND,
      BACKGROUND, BRIGHT, BACKGROUND,
      BACKGROUND, BACKGROUND, BACKGROUND,
    ])

    expect(Array.from(computeBoundaryPixels(m))).toEqual([
      0, 0, 0,
      0, BRIGHT, 0,
      0, 0, 0,
    ])
  })

  it('leaves the interior of a solid region unmarked, only its edge', () => {
    // 5x5, all BRIGHT: the center pixel's 4 neighbors are all BRIGHT too.
    const m = mask(5, 5, new Array(25).fill(BRIGHT))
    const boundary = computeBoundaryPixels(m)

    const centerIndex = 2 * 5 + 2
    expect(boundary[centerIndex]).toBe(0)

    const cornerIndex = 0
    expect(boundary[cornerIndex]).toBe(BRIGHT)
  })

  it('marks both sides of a border between two different non-background classes when each also touches the image edge', () => {
    // Both pixels touch the image edge (counted as background) on their far
    // side, so each is independently marked regardless of the priority rule
    // between BRIGHT and DARK — see the next test for that rule in isolation.
    const m = mask(2, 1, [BRIGHT, DARK])
    expect(Array.from(computeBoundaryPixels(m))).toEqual([BRIGHT, DARK])
  })

  it('gives BRIGHT a complete single-color outline around an interior seam with DARK, while DARK is only outlined where it meets background', () => {
    // 6x5: a BRIGHT block (cols 1-2) next to a DARK block (cols 3-4), rows
    // 1-3, surrounded by a ring of BACKGROUND. Row 2 is the only row not
    // itself touching that ring vertically, isolating the BRIGHT/DARK seam.
    const B = BACKGROUND
    const m = mask(6, 5, [
      B, B,      B,      B,    B,    B,
      B, BRIGHT, BRIGHT, DARK, DARK, B,
      B, BRIGHT, BRIGHT, DARK, DARK, B,
      B, BRIGHT, BRIGHT, DARK, DARK, B,
      B, B,      B,      B,    B,    B,
    ])
    const boundary = computeBoundaryPixels(m)

    // Row 2, interior columns: BRIGHT outranks DARK, so BRIGHT claims the
    // seam between them (col 2) and DARK's side of it (col 3) is left
    // unmarked — a single-color line, not a two-tone one.
    expect(boundary[2 * 6 + 2]).toBe(BRIGHT) // BRIGHT, interior, next to DARK
    expect(boundary[2 * 6 + 3]).toBe(0) // DARK, interior, next to BRIGHT

    // Row 2, outer columns: each touches the background ring on its far
    // side, independent of the seam rule above.
    expect(boundary[2 * 6 + 1]).toBe(BRIGHT)
    expect(boundary[2 * 6 + 4]).toBe(DARK)
  })
})
