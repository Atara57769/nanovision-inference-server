interface UploadFormProps {
  selectedFile: File | null
  onSelectFile: (file: File | null) => void
  onSubmit: () => void
  canSubmit: boolean
  loading: boolean
}

export function UploadForm({
  selectedFile,
  onSelectFile,
  onSubmit,
  canSubmit,
  loading,
}: UploadFormProps) {
  return (
    <form
      className="upload-form"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <label className={loading ? 'dropzone dropzone--disabled' : 'dropzone'}>
        <input
          type="file"
          className="dropzone__input"
          // .tif/.tiff listed explicitly because browsers inconsistently
          // report image/tiff as the file's MIME type in the picker.
          // The real backend on this branch doesn't accept image/tiff yet
          // (support exists on origin/shira/models, not merged here) — a TIFF
          // submitted for real will come back as a 400 invalid_image until it is.
          accept="image/jpeg,image/png,image/webp,image/tiff,.tif,.tiff"
          onChange={(event) => onSelectFile(event.target.files?.[0] ?? null)}
          aria-label="Select image"
          disabled={loading}
        />
        {/* Our own label text, in place of the browser's locale-dependent
            native file-input chrome (e.g. Hebrew on a he-IL system). */}
        <span className="dropzone__text">
          {selectedFile ? selectedFile.name : 'Click to choose a SEM image'}
        </span>
      </label>
      <button className="btn" type="submit" disabled={!canSubmit}>
        {loading ? 'Predicting…' : 'Predict'}
      </button>
    </form>
  )
}
