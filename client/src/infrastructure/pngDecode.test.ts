import { describe, expect, it } from 'vitest'

import { buildGrayscalePngBase64 } from './testFixtures/buildGrayscalePngBase64'
import { decodeMaskPng } from './pngDecode'

describe('decodeMaskPng', () => {
  it('round-trips width, height, and per-pixel class ids', () => {
    const classIds = [0, 1, 2, 1]
    const base64 = buildGrayscalePngBase64(classIds, 2, 2)

    const mask = decodeMaskPng(base64)

    expect(mask.width).toBe(2)
    expect(mask.height).toBe(2)
    expect(Array.from(mask.classGrid)).toEqual(classIds)
  })

  it('handles a larger single-class mask', () => {
    const classIds = new Array(16).fill(2)
    const base64 = buildGrayscalePngBase64(classIds, 4, 4)

    const mask = decodeMaskPng(base64)

    expect(Array.from(mask.classGrid)).toEqual(classIds)
  })
})
