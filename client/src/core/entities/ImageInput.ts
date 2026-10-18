export interface ImageInput {
  // Pinned to ArrayBuffer (not ArrayBufferLike) so this is never backed by a
  // SharedArrayBuffer, which the Blob/FormData APIs this flows into reject.
  bytes: Uint8Array<ArrayBuffer>
  fileName: string
  contentType: string
}
