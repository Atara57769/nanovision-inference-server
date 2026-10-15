import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { SegmentationMask } from '@core/entities/SegmentationMask'

import { MaskCanvas } from './MaskCanvas'

describe('MaskCanvas', () => {
  it('renders a canvas sized to the mask without throwing when getContext is unavailable', () => {
    // jsdom returns null from getContext('2d') by default — this proves the
    // draw effect degrades gracefully instead of crashing the component.
    const mask: SegmentationMask = { width: 3, height: 5, classGrid: new Uint8Array(15) }

    render(<MaskCanvas mask={mask} />)

    const canvas = screen.getByRole('img', { name: 'Segmentation mask' })
    expect(canvas.tagName).toBe('CANVAS')
    expect(canvas).toHaveAttribute('width', '3')
    expect(canvas).toHaveAttribute('height', '5')
  })
})
