"""Value object returned by the inference pipeline."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    """A completed segmentation prediction for one original image.

    ``mask`` is produced by the injected model's postprocessing step.  It has
    the original image's height and width, and stores the model's class IDs:
    background (0), dark (1), and bright (2).
    """

    mask: np.ndarray
    model_name: str
    model_version: str
    inference_time_ms: float
    measurements: dict | None = None
