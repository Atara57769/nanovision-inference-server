import { describe, expect, it } from 'vitest'

import { SEGMENTATION_CLASS } from '@core/entities/SegmentationMask'

import { classColorCss } from './classColors'

describe('classColorCss', () => {
  it('returns the dark-class color', () => {
    expect(classColorCss(SEGMENTATION_CLASS.DARK)).toBe('rgb(99, 61, 191)')
  })

  it('returns the bright-class color', () => {
    expect(classColorCss(SEGMENTATION_CLASS.BRIGHT)).toBe('rgb(250, 204, 21)')
  })

  it('falls back to the fallback color for an unknown class id', () => {
    expect(classColorCss(99)).toBe('rgb(220, 38, 38)')
  })
})
