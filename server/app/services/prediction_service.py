"""Application service for prediction requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from app.models.prediction_result import PredictionResult

if TYPE_CHECKING:
    from app.inference.inference_pipeline import InferencePipeline


class PredictionService:
    """Delegate prediction requests to the inference pipeline."""

    def __init__(self, pipeline: "InferencePipeline") -> None:
        self._pipeline = pipeline

    @property
    def is_ready(self) -> bool:
        """Return whether the inference pipeline is available."""
        return self._pipeline is not None

    def predict(self, image: np.ndarray) -> PredictionResult:
        """Run inference and return the prediction result."""
        return self._pipeline.run(image)