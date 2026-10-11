import numpy as np
import pytest

from app.inference.prediction_result import PredictionResult
from app.services.prediction_service import PredictionService


class FakePipeline:
    def __init__(self, result: PredictionResult) -> None:
        self.result = result
        self.images: list[np.ndarray] = []

    def run(self, image: np.ndarray) -> PredictionResult:
        self.images.append(image)
        return self.result


def test_service_uses_injected_pipeline_once_and_returns_its_exact_result() -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    expected = PredictionResult(np.zeros((3, 4), dtype=np.uint8), "fake", "v1", 1.0)
    pipeline = FakePipeline(expected)

    result = PredictionService(pipeline).predict(image)

    assert len(pipeline.images) == 1
    assert pipeline.images[0] is image
    assert result is expected


def test_service_propagates_pipeline_errors() -> None:
    class FailingPipeline:
        def run(self, image: np.ndarray) -> PredictionResult:
            raise ValueError("invalid image")

    with pytest.raises(ValueError, match="invalid image"):
        PredictionService(FailingPipeline()).predict(np.zeros((1, 1, 3), dtype=np.uint8))
