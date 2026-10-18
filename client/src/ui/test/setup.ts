import '@testing-library/jest-dom/vitest'

// jsdom's Blob/File implementation doesn't provide arrayBuffer() (real
// browsers have supported it for years), so code under test that calls
// file.arrayBuffer() throws here even though it works everywhere this app
// actually runs. Polyfill it via FileReader, which jsdom does implement.
if (typeof Blob.prototype.arrayBuffer !== 'function') {
  Blob.prototype.arrayBuffer = function (this: Blob): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(this)
    })
  }
}

// jsdom doesn't implement the `ImageData` constructor at all — needed by any
// test that exercises real canvas-paint code (not just a stubbed context).
if (typeof globalThis.ImageData === 'undefined') {
  class ImageDataPolyfill {
    readonly data: Uint8ClampedArray
    readonly width: number
    readonly height: number
    readonly colorSpace = 'srgb' as const

    constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, height?: number) {
      const width = typeof dataOrWidth === 'number' ? dataOrWidth : widthOrHeight
      const resolvedHeight = typeof dataOrWidth === 'number' ? widthOrHeight : (height as number)
      if (width <= 0 || resolvedHeight <= 0) {
        throw new DOMException(
          'The source width and height must be greater than zero.',
          'IndexSizeError',
        )
      }
      this.width = width
      this.height = resolvedHeight
      this.data =
        typeof dataOrWidth === 'number'
          ? new Uint8ClampedArray(width * resolvedHeight * 4)
          : dataOrWidth
    }
  }
  // @ts-expect-error -- minimal test-only stand-in for the DOM's real ImageData
  globalThis.ImageData = ImageDataPolyfill
}

// jsdom doesn't implement object URLs either.
if (typeof URL.createObjectURL !== 'function') {
  URL.createObjectURL = () => 'blob:mock-url'
}
if (typeof URL.revokeObjectURL !== 'function') {
  URL.revokeObjectURL = () => {}
}
