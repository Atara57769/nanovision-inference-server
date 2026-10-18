import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { UploadForm } from './UploadForm'

describe('UploadForm', () => {
  it('disables submit when canSubmit is false', () => {
    render(
      <UploadForm
        selectedFile={null}
        onSelectFile={vi.fn()}
        onSubmit={vi.fn()}
        canSubmit={false}
        loading={false}
      />,
    )
    expect(screen.getByRole('button', { name: 'Predict' })).toBeDisabled()
  })

  it('enables submit and shows the selected file name when canSubmit is true', () => {
    const file = new File(['x'], 'cells.png', { type: 'image/png' })
    render(
      <UploadForm
        selectedFile={file}
        onSelectFile={vi.fn()}
        onSubmit={vi.fn()}
        canSubmit={true}
        loading={false}
      />,
    )
    expect(screen.getByRole('button', { name: 'Predict' })).toBeEnabled()
    expect(screen.getByText('cells.png')).toBeInTheDocument()
  })

  it('shows a loading label and calls onSubmit when clicked', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    const file = new File(['x'], 'cells.png', { type: 'image/png' })
    render(
      <UploadForm
        selectedFile={file}
        onSelectFile={vi.fn()}
        onSubmit={onSubmit}
        canSubmit={true}
        loading={true}
      />,
    )
    const button = screen.getByRole('button', { name: 'Predicting…' })
    expect(button).toBeEnabled()
    await user.click(button)
    expect(onSubmit).toHaveBeenCalledOnce()
  })

  it('calls onSelectFile with the chosen file', async () => {
    const user = userEvent.setup()
    const onSelectFile = vi.fn()
    render(
      <UploadForm
        selectedFile={null}
        onSelectFile={onSelectFile}
        onSubmit={vi.fn()}
        canSubmit={false}
        loading={false}
      />,
    )
    const file = new File(['x'], 'cells.png', { type: 'image/png' })
    const input = screen.getByLabelText('Select image')
    await user.upload(input, file)
    expect(onSelectFile).toHaveBeenCalledWith(file)
  })
})
