import * as UTIF from 'utif2'
import { describe, expect, it } from 'vitest'

import { decodeTiffPreview } from './decodeTiffPreview'

function buildTiffFile(width: number, height: number, rgba: number[]): File {
  const buffer = UTIF.encodeImage(Uint8Array.from(rgba), width, height)
  return new File([buffer], 'sample.tif', { type: 'image/tiff' })
}

describe('decodeTiffPreview', () => {
  it('round-trips a small TIFF into pixel data with matching dimensions and colors', async () => {
    // 2x1 image: one red pixel, one green pixel, both opaque.
    const file = buildTiffFile(2, 1, [255, 0, 0, 255, 0, 255, 0, 255])

    const decoded = await decodeTiffPreview(file)

    expect(decoded).not.toBeNull()
    expect(decoded!.width).toBe(2)
    expect(decoded!.height).toBe(1)
    expect(Array.from(decoded!.data.slice(0, 4))).toEqual([255, 0, 0, 255])
    expect(Array.from(decoded!.data.slice(4, 8))).toEqual([0, 255, 0, 255])
  })

  it('resolves null instead of throwing for a non-TIFF file', async () => {
    const file = new File(['not a tiff'], 'bad.tif', { type: 'image/tiff' })

    await expect(decodeTiffPreview(file)).resolves.toBeNull()
  })
})
