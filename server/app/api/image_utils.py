import io
import logging

import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
_READ_CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)


class InvalidImageError(Exception):
    pass


def validate_content_type(content_type: str | None) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(f"Unsupported content type: {content_type!r}")


def validate_image(content_type: str | None, content: bytes) -> None:
    """Self-contained: re-checks content type plus the checks that need the full body."""
    validate_content_type(content_type)
    if len(content) == 0:
        raise InvalidImageError("Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError("Uploaded file exceeds the 10MB size limit")


async def read_upload_within_limit(file: UploadFile) -> bytes:
    """Read the upload in chunks, aborting as soon as MAX_FILE_SIZE_BYTES is exceeded
    instead of buffering the whole body first."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE_BYTES:
            raise InvalidImageError("Uploaded file exceeds the 10MB size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_image(content: bytes) -> np.ndarray:
    """Decode to an (H, W, 3) uint8 RGB array — the shape/dtype PredictionService.predict() expects."""
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        logger.exception("Failed to decode uploaded image")
        raise InvalidImageError("Could not decode image") from exc
