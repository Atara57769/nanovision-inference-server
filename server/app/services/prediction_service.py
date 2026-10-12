"""Application service for prediction requests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from app.models.prediction_result import PredictionResult

if TYPE_CHECKING:
    from app.inference.inference_pipeline import InferencePipeline


class PredictionService:
    """Delegate a prediction request to an injected inference pipeline."""

    def __init__(self, pipeline: "InferencePipeline") -> None:
        self._pipeline = pipeline

    @property
    def is_ready(self) -> bool:
        """Whether a real pipeline/model is loaded, not just whether this object exists.

        Assumption to confirm with Part 2's author: the real PredictionService should
        expose this same property once it's merged, so /health keeps working unchanged.
        """
        return self._pipeline is not None

    def predict(self, image: np.ndarray) -> PredictionResult:
        """Return the pipeline result, allowing failures to reach the API."""

        return self._pipeline.run(image)
