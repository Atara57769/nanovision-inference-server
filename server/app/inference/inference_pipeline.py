"""Model-agnostic inference orchestration."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.models.prediction_result import PredictionResult

if TYPE_CHECKING:
    from app.models.base_model import BaseModel


class InferencePipeline:
    """Run an already-loaded model's complete inference workflow.

    The pipeline deliberately knows nothing about image transformations or
    model architecture.  The injected model owns preprocessing, batching,
    prediction interpretation, merging, and postprocessing.
    """

    def __init__(self, model: "BaseModel") -> None:
        self._model = model

    def run(self, image: np.ndarray) -> PredictionResult:
        """Infer a class mask for an RGB image using the injected model.

        ``preprocess`` must return an object with ``batches`` and ``metadata``
        attributes.  The pipeline passes that metadata through unchanged to
        ``postprocess`` so the model can reconstruct the original image.
        """

        started_at = perf_counter()

        # The model is loaded and set to evaluation mode by the composition
        # root.  Do not load or reconfigure it on a request path.
        with torch.inference_mode():
            prepared = self._model.preprocess(image)
            predictions = [self._model.predict(batch) for batch in prepared.batches]
            mask = self._model.postprocess(predictions, prepared.metadata)

        model_info = self._model.info()
        elapsed_ms = (perf_counter() - started_at) * 1_000

        return PredictionResult(
            mask=mask,
            model_name=self._model_info_value(model_info, "name", "model_name"),
            model_version=self._model_info_value(model_info, "version", "model_version"),
            inference_time_ms=elapsed_ms,
        )

    @staticmethod
    def _model_info_value(model_info: Any, *fields: str) -> str:
        """Read a named value from the project ModelInfo object.

        Supporting a mapping as well as attribute access keeps this boundary
        compatible with lightweight test doubles without coupling it to a
        particular model implementation.
        """

        for field in fields:
            if isinstance(model_info, dict) and field in model_info:
                return str(model_info[field])
            if not isinstance(model_info, dict) and hasattr(model_info, field):
                return str(getattr(model_info, field))
        raise AttributeError(f"ModelInfo must define one of: {', '.join(fields)}")
