import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SegmentationMask } from '@core/entities/SegmentationMask'

import { MaskCanvas } from './MaskCanvas'

type FakeCall = { method: 'clearRect' | 'drawImage' | 'putImageData' }

/** A working (but inert) 2D context, since jsdom's own getContext('2d')
 *  always returns null — without this, none of MaskCanvas's actual drawing
 *  code (drawBackground/paintOutline/buildOutlineImageData) ever runs under
 *  test. Every canvas the component creates (the visible one plus the
 *  offscreen background/outline ones) shares this one fake, recording calls
 *  into a single shared log so call order across them can be asserted. */
function stubCanvasContext(calls: FakeCall[]) {
  const fakeCtx = {
    clearRect: () => calls.push({ method: 'clearRect' }),
    drawImage: () => calls.push({ method: 'drawImage' }),
    putImageData: () => calls.push({ method: 'putImageData' }),
  }
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
    (() => fakeCtx) as unknown as typeof HTMLCanvasElement.prototype.getContext,
  )
}

/** Stands in for the real `Image`, whose `onload` only ever fires
 *  asynchronously — this lets the test control exactly when that happens. */
class FakeImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor() {
    createdImages.push(this)
  }
  set src(_value: string) {
    // no-op: the test triggers onload/onerror itself
  }
}
let createdImages: FakeImage[] = []

// A couple of these cover the plain decode paths, but the TIFF-fallback path
// (decodeTiffPreview -> file.arrayBuffer(), polyfilled via FileReader) needs
// several real event-loop turns, not microtasks — one tick alone isn't
// enough to flush it.
async function flushMicrotasks(turns = 5): Promise<void> {
  for (let i = 0; i < turns; i++) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

const mask: SegmentationMask = { width: 3, height: 5, classGrid: new Uint8Array(15) }

describe('MaskCanvas', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    createdImages = []
  })

  it('renders a canvas sized to the mask without throwing when getContext is unavailable', () => {
    // jsdom returns null from getContext('2d') by default — this proves the
    // draw effect degrades gracefully instead of crashing the component.
    render(<MaskCanvas mask={mask} />)

    const canvas = screen.getByRole('img', { name: 'Segmentation mask' })
    expect(canvas.tagName).toBe('CANVAS')
    expect(canvas).toHaveAttribute('width', '3')
    expect(canvas).toHaveAttribute('height', '5')
  })

  it('does not crash on a degenerate 0x0 mask, even with a working canvas', async () => {
    stubCanvasContext([])
    const emptyMask: SegmentationMask = { width: 0, height: 0, classGrid: new Uint8Array(0) }

    render(<MaskCanvas mask={emptyMask} />)
    await act(flushMicrotasks)

    expect(screen.getByRole('img', { name: 'Segmentation mask' })).toBeInTheDocument()
  })

  it('paints background and outline together, never the outline alone on a blank canvas', async () => {
    const calls: FakeCall[] = []
    stubCanvasContext(calls)
    vi.stubGlobal('Image', FakeImage)

    const sourceImage = new File(['bytes'], 'cells.png', { type: 'image/png' })
    render(<MaskCanvas mask={mask} sourceImage={sourceImage} />)

    // The native image decode is asynchronous (a real <img> never loads
    // synchronously) — nothing should be painted yet.
    expect(calls).toHaveLength(0)
    expect(createdImages).toHaveLength(1)

    await act(async () => {
      createdImages[0].onload?.()
      await flushMicrotasks()
    })

    // Exactly one paint pass happened (one clearRect), and it included both
    // the background and the outline — never an earlier pass with the
    // outline alone on an empty background.
    expect(calls.filter((call) => call.method === 'clearRect')).toHaveLength(1)
    expect(calls.filter((call) => call.method === 'drawImage').length).toBeGreaterThanOrEqual(2)
  })

  it("shows the fallback message when the uploaded file can't be previewed", async () => {
    stubCanvasContext([])
    vi.stubGlobal('Image', FakeImage)

    // Not a real image or TIFF — both the native decode and the TIFF
    // fallback fail, so neither can produce a background.
    const sourceImage = new File(['not an image'], 'notes.txt', { type: 'text/plain' })
    render(<MaskCanvas mask={mask} sourceImage={sourceImage} />)

    await act(async () => {
      createdImages[0].onerror?.()
      await flushMicrotasks()
    })

    expect(
      screen.getByText("This browser can't preview that file type — showing the outline only."),
    ).toBeInTheDocument()
  })
})
