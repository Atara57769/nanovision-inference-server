import { encode } from 'upng-js'
import { describe, expect, it } from 'vitest'

import { decodeMaskPng } from './pngDecode'

// @types/upng-js doesn't declare the 6th `forbidPlte` parameter, but the
// runtime accepts it — it's needed to keep the fixture an ctype=0 grayscale
// PNG (what PIL actually emits for a 2D uint8 array), matching the real
// server's output instead of whatever indexed-palette encoding UPNG's own
// heuristics would otherwise pick for a low-color-count test image.
const encodeForbiddingPalette = encode as (
  imgs: ArrayBuffer[],
  w: number,
  h: number,
  cnum: number,
  dels: number[] | undefined,
  forbidPlte: boolean,
) => ArrayBuffer

function buildGrayscalePngBase64(
  classIds: number[],
  width: number,
  height: number,
): string {
  const rgba = new Uint8Array(width * height * 4)
  for (let i = 0; i < classIds.length; i++) {
    const value = classIds[i]
    rgba[i * 4] = value
    rgba[i * 4 + 1] = value
    rgba[i * 4 + 2] = value
    rgba[i * 4 + 3] = 255
  }

  const pngBytes = new Uint8Array(
    encodeForbiddingPalette([rgba.buffer], width, height, 0, undefined, true),
  )
  let binary = ''
  for (const byte of pngBytes) {
    binary += String.fromCharCode(byte)
  }
  return btoa(binary)
}

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
