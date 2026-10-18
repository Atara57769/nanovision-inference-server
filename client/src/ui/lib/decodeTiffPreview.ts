import * as UTIF from 'utif2'

/** Plain RGBA pixel data — not `ImageData` itself, so this stays testable
 *  without a canvas (jsdom doesn't implement the `ImageData` constructor). */
export interface DecodedImage {
  width: number
  height: number
  // Pinned to ArrayBuffer (not ArrayBufferLike), matching ImageData's own
  // constructor — see the same fix on core/entities/ImageInput.ts.
  data: Uint8ClampedArray<ArrayBuffer>
}

/**
 * Decodes a TIFF file into RGBA pixel data for preview — browsers can't
 * decode TIFF through <img>/Image() the way they do PNG/JPEG/WebP, so this
 * is the fallback used only when that native path fails. Returns null for
 * anything that isn't a decodable TIFF (corrupt file, empty page list, etc.).
 */
export async function decodeTiffPreview(file: File): Promise<DecodedImage | null> {
  try {
    const buffer = await file.arrayBuffer()
    const ifds = UTIF.decode(buffer)
    const page = ifds[0]
    if (!page) {
      return null
    }
    // decodeImage() is what actually sets page.width/page.height — decode()
    // alone only parses tags. UTIF doesn't throw on non-TIFF bytes either
    // way, so a usable width/height is checked explicitly afterward.
    UTIF.decodeImage(buffer, page)
    if (!page.width || !page.height) {
      return null
    }
    const rgba = UTIF.toRGBA8(page)
    // The Uint8ClampedArray(typedArray) constructor already copies values
    // into a fresh ArrayBuffer rather than aliasing rgba's own buffer,
    // matching ImageData's constructor.
    const data = new Uint8ClampedArray(rgba)
    return { width: page.width, height: page.height, data }
  } catch {
    return null
  }
}
