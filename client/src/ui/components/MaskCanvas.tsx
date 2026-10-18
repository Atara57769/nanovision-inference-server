import { useEffect, useRef, useState } from 'react'

import { computeBoundaryPixels } from '@core/policies/computeBoundaryPixels'
import type { SegmentationMask } from '@core/entities/SegmentationMask'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

import { CLASS_COLORS, FALLBACK_CLASS_COLOR } from '../lib/classColors'
import { decodeTiffPreview } from '../lib/decodeTiffPreview'

function buildOutlineImageData(mask: SegmentationMask): ImageData | null {
  // The `ImageData` constructor throws for a zero dimension — a degenerate
  // mask (e.g. an empty model/server response) must not crash the canvas.
  if (mask.width === 0 || mask.height === 0) {
    return null
  }

  const boundary = computeBoundaryPixels(mask)
  const outline = new ImageData(mask.width, mask.height)
  for (let i = 0; i < boundary.length; i++) {
    const classId = boundary[i]
    if (classId === SEGMENTATION_CLASS.BACKGROUND) {
      continue
    }
    const [r, g, b] = CLASS_COLORS[classId] ?? FALLBACK_CLASS_COLOR
    outline.data[i * 4] = r
    outline.data[i * 4 + 1] = g
    outline.data[i * 4 + 2] = b
    outline.data[i * 4 + 3] = 255
  }
  return outline
}

/** Native decode first (fast path for PNG/JPEG/WebP, anything the browser
 *  already knows). Browsers can't decode TIFF this way, so that's the one
 *  case the onerror branch below falls through to our own TIFF decoder for. */
function loadNativeImage(objectUrl: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => resolve(null)
    image.src = objectUrl
  })
}

type BackgroundSource =
  | { kind: 'none' }
  | { kind: 'unavailable' }
  | { kind: 'image'; image: HTMLImageElement }
  | { kind: 'raw'; width: number; height: number; data: Uint8ClampedArray<ArrayBuffer> }

/** Decides what to draw behind the outline for a given upload, if anything.
 *  Resolves once — the caller only paints after this settles, so background
 *  and outline always appear together, never outline-first on a blank canvas. */
async function loadBackground(sourceImage: File | null | undefined): Promise<BackgroundSource> {
  if (!sourceImage) {
    return { kind: 'none' }
  }

  const objectUrl = URL.createObjectURL(sourceImage)
  const nativeImage = await loadNativeImage(objectUrl).finally(() =>
    URL.revokeObjectURL(objectUrl),
  )
  if (nativeImage) {
    return { kind: 'image', image: nativeImage }
  }

  const decoded = await decodeTiffPreview(sourceImage)
  return decoded ? { kind: 'raw', ...decoded } : { kind: 'unavailable' }
}

function imageDataToCanvas(data: ImageData): HTMLCanvasElement | null {
  const canvas = document.createElement('canvas')
  canvas.width = data.width
  canvas.height = data.height
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return null
  }
  ctx.putImageData(data, 0, 0)
  return canvas
}

function drawBackground(
  ctx: CanvasRenderingContext2D,
  source: BackgroundSource,
  width: number,
  height: number,
): void {
  if (source.kind === 'image') {
    ctx.drawImage(source.image, 0, 0, width, height)
    return
  }
  if (source.kind === 'raw') {
    const bgCanvas = imageDataToCanvas(new ImageData(source.data, source.width, source.height))
    if (bgCanvas) {
      ctx.drawImage(bgCanvas, 0, 0, width, height)
    }
  }
}

function paintOutline(ctx: CanvasRenderingContext2D, mask: SegmentationMask): void {
  const outlineData = buildOutlineImageData(mask)
  if (!outlineData) {
    return
  }
  // A separate canvas for the outline, composited in via drawImage —
  // putImageData writes raw pixels (including transparent ones) and
  // would erase the source image already drawn underneath.
  const outlineCanvas = imageDataToCanvas(outlineData)
  if (outlineCanvas) {
    ctx.drawImage(outlineCanvas, 0, 0)
  }
}

interface MaskCanvasProps {
  mask: SegmentationMask
  /** The originally uploaded image, drawn underneath the outline when given. */
  sourceImage?: File | null
}

export function MaskCanvas({ mask, sourceImage }: MaskCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [backgroundUnavailable, setBackgroundUnavailable] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    // jsdom (unit tests) has no real 2D canvas context — skip the decode and
    // draw work entirely rather than loading a background nothing can paint.
    if (!canvas || !ctx) {
      return
    }

    let cancelled = false

    void loadBackground(sourceImage).then((background) => {
      if (cancelled) {
        return
      }
      // Background and outline are only ever painted together, once the
      // background decision is final — never the outline alone first.
      setBackgroundUnavailable(background.kind === 'unavailable')
      ctx.clearRect(0, 0, mask.width, mask.height)
      drawBackground(ctx, background, mask.width, mask.height)
      paintOutline(ctx, mask)
    })

    return () => {
      cancelled = true
    }
  }, [mask, sourceImage])

  return (
    <div className="mask-canvas">
      <canvas
        ref={canvasRef}
        width={mask.width}
        height={mask.height}
        role="img"
        aria-label="Segmentation mask"
      />
      {backgroundUnavailable && (
        <p>This browser can't preview that file type — showing the outline only.</p>
      )}
    </div>
  )
}
