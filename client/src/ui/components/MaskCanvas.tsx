import { useEffect, useRef, useState } from 'react'

import { computeBoundaryPixels } from '@core/policies/computeBoundaryPixels'
import type { SegmentationMask } from '@core/entities/SegmentationMask'
import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

import { decodeTiffPreview } from '../lib/decodeTiffPreview'

type Rgb = readonly [number, number, number]

const CLASS_COLORS: Record<number, Rgb> = {
  [SEGMENTATION_CLASS.BRIGHT]: [250, 204, 21],
  [SEGMENTATION_CLASS.DARK]: [99, 61, 191],
}
const FALLBACK_COLOR: Rgb = [220, 38, 38]

function buildOutlineImageData(mask: SegmentationMask): ImageData {
  const boundary = computeBoundaryPixels(mask)
  const outline = new ImageData(mask.width, mask.height)
  for (let i = 0; i < boundary.length; i++) {
    const classId = boundary[i]
    if (classId === SEGMENTATION_CLASS.BACKGROUND) {
      continue
    }
    const [r, g, b] = CLASS_COLORS[classId] ?? FALLBACK_COLOR
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

interface MaskCanvasProps {
  mask: SegmentationMask
  /** The originally uploaded image, drawn underneath the outline when given. */
  sourceImage?: File | null
}

export function MaskCanvas({ mask, sourceImage }: MaskCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Set when neither the native decode nor the TIFF fallback could produce a
  // background — without this, that case used to leave the canvas blank
  // with no outline either, looking identical to "nothing happened".
  const [backgroundUnavailable, setBackgroundUnavailable] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    // jsdom (unit tests) has no real 2D canvas context — draw only when one exists.
    if (!canvas || !ctx) {
      return
    }

    let cancelled = false
    let objectUrl: string | null = null
    setBackgroundUnavailable(false)

    function paintOutline() {
      // A separate canvas for the outline, composited in via drawImage —
      // putImageData writes raw pixels (including transparent ones) and
      // would erase the source image already drawn underneath.
      const outlineCanvas = document.createElement('canvas')
      outlineCanvas.width = mask.width
      outlineCanvas.height = mask.height
      const outlineCtx = outlineCanvas.getContext('2d')
      if (!outlineCtx || !ctx) {
        return
      }
      outlineCtx.putImageData(buildOutlineImageData(mask), 0, 0)
      ctx.drawImage(outlineCanvas, 0, 0)
    }

    function drawRgbaBackground(width: number, height: number, data: Uint8ClampedArray<ArrayBuffer>) {
      const bgCanvas = document.createElement('canvas')
      bgCanvas.width = width
      bgCanvas.height = height
      const bgCtx = bgCanvas.getContext('2d')
      if (!bgCtx || !ctx) {
        return
      }
      bgCtx.putImageData(new ImageData(data, width, height), 0, 0)
      ctx.drawImage(bgCanvas, 0, 0, mask.width, mask.height)
    }

    async function draw() {
      if (!sourceImage) {
        ctx!.clearRect(0, 0, mask.width, mask.height)
        paintOutline()
        return
      }

      objectUrl = URL.createObjectURL(sourceImage)
      const nativeImage = await loadNativeImage(objectUrl)
      if (cancelled || !ctx) {
        return
      }

      if (nativeImage) {
        ctx.drawImage(nativeImage, 0, 0, mask.width, mask.height)
      } else {
        const decoded = await decodeTiffPreview(sourceImage)
        if (cancelled || !ctx) {
          return
        }
        if (decoded) {
          drawRgbaBackground(decoded.width, decoded.height, decoded.data)
        } else {
          setBackgroundUnavailable(true)
          ctx.clearRect(0, 0, mask.width, mask.height)
        }
      }
      paintOutline()
    }

    void draw()

    return () => {
      cancelled = true
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [mask, sourceImage])

  return (
    <div>
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
