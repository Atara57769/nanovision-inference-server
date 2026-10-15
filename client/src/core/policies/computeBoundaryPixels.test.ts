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

  it('marks both sides of a border between two different non-background classes', () => {
    const m = mask(2, 1, [BRIGHT, DARK])
    expect(Array.from(computeBoundaryPixels(m))).toEqual([BRIGHT, DARK])
  })
})
