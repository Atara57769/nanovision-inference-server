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
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <input
        type="file"
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
      {selectedFile && <p>{selectedFile.name}</p>}
      <button type="submit" disabled={!canSubmit}>
        {loading ? 'Predicting…' : 'Predict'}
      </button>
    </form>
  )
}
