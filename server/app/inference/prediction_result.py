# STUB — mirrors the real contract confirmed on the `prediction-inference-pipeline` branch
# (server/app/inference/prediction_result.py there). Delete this once that branch is merged
# into main; field names/types below must stay identical until then.
from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    mask: np.ndarray  # (H, W) uint8 array of class IDs
    model_name: str
    model_version: str
    inference_time_ms: float
