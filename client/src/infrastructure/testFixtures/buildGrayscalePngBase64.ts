import { encode } from 'upng-js'

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

/** Builds a base64 grayscale PNG whose pixel values are the given class ids
 *  — a stand-in for the server's `mask` field, shared by tests that need a
 *  real decodable mask rather than mocking decodeMaskPng itself. */
export function buildGrayscalePngBase64(
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
