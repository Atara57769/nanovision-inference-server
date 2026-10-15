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
