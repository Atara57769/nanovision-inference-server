import { decode, toRGBA8 } from 'upng-js'

import type { SegmentationMask } from '@core/entities/SegmentationMask'

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes.buffer
}

/** The server encodes the mask as a grayscale PNG where each pixel's value IS the class id. */
export function decodeMaskPng(base64Png: string): SegmentationMask {
  const image = decode(base64ToArrayBuffer(base64Png))
  const [rgba] = toRGBA8(image)
  const rgbaBytes = new Uint8Array(rgba)

  const classGrid = new Uint8Array(image.width * image.height)
  for (let i = 0; i < classGrid.length; i++) {
    classGrid[i] = rgbaBytes[i * 4]
  }

  return { width: image.width, height: image.height, classGrid }
}
