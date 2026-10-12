"""Shared test fixtures."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
import torch

from app.models.classes import CLASS_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture
def write_checkpoint(tmp_path: Path) -> Callable[..., Path]:
    """Save weights in the research trainer's checkpoint format (NanoVision src/train.py).

    Tests never use real checkpoints: the repository is public, and a checkpoint
    written here lets each test choose exactly what the file claims about itself.
    """

    def write(
        weights: Mapping[str, torch.Tensor],
        *,
        model_name: str,
        classes: Sequence[str] = CLASS_NAMES,
        crop_size: int = 256,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        step: int = 100,
        file_name: str = "best.pt",
    ) -> Path:
        path = tmp_path / file_name
        torch.save(
            {
                "model": dict(weights),
                "optimizer": {},
                "step": step,
                "best_score": 0.5,
                "config": {
                    "model": {"name": model_name},
                    "data": {
                        "classes": list(classes),
                        "crop_size": crop_size,
                        "input": {"channels": 3, "mean": list(mean), "std": list(std)},
                    },
                },
                "label_version": "v1",
                # str(), as train.py does: weights_only loading rejects the TorchVersion class.
                "torch_version": str(torch.__version__),
            },
            path,
        )
        return path

    return write
