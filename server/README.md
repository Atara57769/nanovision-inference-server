# NanoVision inference server

Segments SEM images of Cu–Ag particles. Send an image, get back a mask of the
same size: `0` background, `1` dark particle part, `2` bright particle part.
The served model is U-Net++ with a ResNet34 encoder, trained by team AMAT-2.

## Setup

Run everything from this `server/` folder.

```bash
python -m venv ../.venv
../.venv/Scripts/activate            # Linux/macOS: source ../.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

At first startup the server downloads the official U-Net++ checkpoint from the
[NanoVision Hugging Face repository](https://huggingface.co/atara57769/nanovision-segmentation),
then verifies its SHA-256 and caches it at `checkpoints/unetpp/unetplusplus_best.pt`.
It is not in git: the repository is public, and `checkpoints/` is ignored. To
run without network access, put that verified file in the cache path first, or
set `NANOVISION_CHECKPOINT` to a local copy.

## Run

```bash
uvicorn app.main:app --port 8000
```

The model is loaded once at startup. If it cannot be loaded (missing or
incompatible checkpoint, invalid config), the server still starts but is not
ready, and the log says why.

- `GET /health` returns `{"status": "ok", "service_ready": true}` when the model is loaded.
- `GET /ready` is a readiness probe: it returns 200 only after the model is loaded, otherwise 503.
- `POST /predict` takes a TIFF, PNG, JPEG or WebP file (up to 10 MB) as form field
  `file` and returns the mask as a base64 PNG, with `model_name`, `model_version`
  and `inference_time_ms`. It answers 503 while the server is not ready.

```bash
curl -F "file=@2Eu1_005.tif;type=image/tiff" http://127.0.0.1:8000/predict
```

On a laptop CPU, one 1024×768 image takes about 20–25 seconds.

## Configuration

`app/config/model_config.yaml` chooses the model and how it runs: `active_model`,
`device` (`cpu`, `cuda` or `auto`), `batch_size`, `footer_height` (rows of the
microscope's info bar, returned as background) and, per model, `checkpoint`,
`tile_size` and `overlap`. The shipped configuration also pins the Hugging Face
artifact URL and SHA-256. Changing the model needs only this file and a restart.

Two optional environment variables override it, for example on another machine:

- `NANOVISION_MODEL_CONFIG`: path to a different config file
- `NANOVISION_CHECKPOINT`: path to the active model's checkpoint

Relative paths, in the file and in these variables, start at this `server/` folder.

## Tests

```bash
python -m pytest
```

The tests never use the real checkpoint; they build small or untrained models.

## Verified against research

On 2026-10-13 the server, end to end (TIFF decoding, config, model loading,
inference pipeline), was run on the 29 DEV images with the official checkpoint
and scored with the research evaluator (NanoVision `code/amat-2/src/evaluate.py`
@ d8f35ed). It reproduced the reported DEV score exactly: 0.8537261284. On three
images its masks were also compared with research `predict_mask`: identical in
every pixel.
