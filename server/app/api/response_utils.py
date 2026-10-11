import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from app.inference.prediction_result import PredictionResult


def encode_mask(mask: np.ndarray) -> str:
    mask_image = Image.fromarray(mask.astype(np.uint8))
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def serialize_result(result: PredictionResult) -> dict[str, Any]:
    return {
        "mask": encode_mask(result.mask),
        "model_name": result.model_name,
        "model_version": result.model_version,
        "inference_time_ms": result.inference_time_ms,
    }
