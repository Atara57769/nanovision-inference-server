# STUB — mirrors the real contract confirmed on the `prediction-inference-pipeline` branch
# (server/app/services/prediction_service.py there, which delegates to InferencePipeline.run()).
# Delete this once that branch is merged into main.
import time

import numpy as np

from app.inference.prediction_result import PredictionResult


class PredictionService:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    @property
    def is_ready(self) -> bool:
        """Whether a real pipeline/model is loaded, not just whether this object exists.

        Assumption to confirm with Part 2's author: the real PredictionService should
        expose this same property once it's merged, so /health keeps working unchanged.
        """
        return self._pipeline is not None

    def predict(self, image: np.ndarray) -> PredictionResult:
        start = time.perf_counter()
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return PredictionResult(
            mask=mask,
            model_name="stub",
            model_version="stub",
            inference_time_ms=elapsed_ms,
        )
