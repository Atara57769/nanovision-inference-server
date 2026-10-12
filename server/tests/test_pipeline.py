from dataclasses import dataclass

import numpy as np
import pytest
import torch

from app.inference.inference_pipeline import InferencePipeline
from app.models.prediction_result import PredictionResult


@dataclass
class FakePreprocessResult:
    batches: list[torch.Tensor]
    metadata: object


@dataclass
class FakeModelInfo:
    name: str = "fake-segmenter"
    version: str = "test-v1"


class FakeModel:
    def __init__(self, batches: list[torch.Tensor], metadata: object) -> None:
        self.prepared = FakePreprocessResult(batches=batches, metadata=metadata)
        self.preprocess_images: list[np.ndarray] = []
        self.predicted_batches: list[torch.Tensor] = []
        self.inference_mode_states: list[bool] = []
        self.postprocess_arguments: tuple[list[torch.Tensor], object] | None = None
        self.info_calls = 0

    def preprocess(self, image: np.ndarray) -> FakePreprocessResult:
        self.preprocess_images.append(image)
        return self.prepared

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        self.predicted_batches.append(batch)
        self.inference_mode_states.append(torch.is_inference_mode_enabled())
        return batch + 1

    def postprocess(self, predictions: list[torch.Tensor], metadata: object) -> np.ndarray:
        self.postprocess_arguments = (predictions, metadata)
        return np.full(metadata["original_shape"], 2, dtype=np.uint8)

    def info(self) -> FakeModelInfo:
        self.info_calls += 1
        return FakeModelInfo()


def test_pipeline_delegates_each_step_and_returns_complete_result() -> None:
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    batches = [torch.zeros((1, 3, 2, 2)), torch.ones((1, 3, 2, 2))]
    metadata = {"original_shape": image.shape[:2], "request_id": "example"}
    model = FakeModel(batches, metadata)

    result = InferencePipeline(model).run(image)

    assert len(model.preprocess_images) == 1
    assert model.preprocess_images[0] is image
    assert len(model.predicted_batches) == len(batches)
    assert all(actual is expected for actual, expected in zip(model.predicted_batches, batches))
    assert model.inference_mode_states == [True, True]
    assert model.postprocess_arguments is not None
    predictions, passed_metadata = model.postprocess_arguments
    assert passed_metadata is metadata
    assert torch.equal(predictions[0], batches[0] + 1)
    assert torch.equal(predictions[1], batches[1] + 1)
    assert isinstance(result, PredictionResult)
    assert result.mask.shape == image.shape[:2]
    assert np.all(result.mask == 2)
    assert result.model_name == "fake-segmenter"
    assert result.model_version == "test-v1"
    assert result.inference_time_ms >= 0
    assert model.info_calls == 1


def test_pipeline_propagates_model_errors_without_reloading() -> None:
    class FailingModel(FakeModel):
        def predict(self, batch: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("prediction failed")

    model = FailingModel([torch.zeros((1, 3, 2, 2))], {"original_shape": (2, 2)})

    with pytest.raises(RuntimeError, match="prediction failed"):
        InferencePipeline(model).run(np.zeros((2, 2, 3), dtype=np.uint8))

    # Inference calls no loading API; the only model interactions before the
    # failure are preprocessing and prediction.
    assert len(model.preprocess_images) == 1
    assert model.info_calls == 0
