import io

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class InvalidImageError(Exception):
    pass


def validate_image(content_type: str | None, content: bytes) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(f"Unsupported content type: {content_type!r}")
    if len(content) == 0:
        raise InvalidImageError("Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise InvalidImageError("Uploaded file exceeds the 10MB size limit")


def decode_image(content: bytes) -> np.ndarray:
    """Decode to an (H, W, 3) uint8 RGB array — the shape/dtype PredictionService.predict() expects."""
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    return np.array(image.convert("RGB"), dtype=np.uint8)
