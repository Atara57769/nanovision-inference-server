# STUB — mirrors the real contract confirmed on the `prediction-inference-pipeline` branch
# (server/app/services/prediction_service.py there, which delegates to InferencePipeline.run()).
# Delete this once that branch is merged into main.
import time

import numpy as np

from app.inference.prediction_result import PredictionResult


class PredictionService:
    def __init__(self, pipeline):
        self._pipeline = pipeline

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
