"""Read a checkpoint saved by the NanoVision research trainer.

Format: NanoVision code/amat-2/src/train.py ``save_checkpoint`` @ d8f35ed --
``{"model": state_dict, "optimizer": ..., "step": int, "best_score": float,
"config": <the full training config>, "label_version": str, "torch_version": str}``.

This module is the only place that knows those key names. The rest of the
server works with the typed ``CheckpointMetadata``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

# Enough of the file hash to tell two checkpoints apart in a version string.
_VERSION_HASH_LENGTH = 8


@dataclass(frozen=True)
class CheckpointMetadata:
    """What a checkpoint says about how its weights were trained."""

    model_name: str
    class_names: tuple[str, ...]
    crop_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    label_version: str
    step: int
    best_score: float
    sha256: str

    @property
    def version(self) -> str:
        """Readable and unique: label version, training step and the file's own hash.

        The hash matters because two training runs can both stop at the same step;
        it proves exactly which weights produced a mask.
        """

        return f"{self.label_version}-step{self.step}-{self.sha256[:_VERSION_HASH_LENGTH]}"


def load_checkpoint(path: Path) -> tuple[Mapping[str, torch.Tensor], CheckpointMetadata]:
    """Return the network weights and the training metadata of one checkpoint file."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    # weights_only=True refuses to run code hidden in the file: a checkpoint is
    # data, and loading one must never be able to execute anything.
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        config = state["config"]
        data = config["data"]
        metadata = CheckpointMetadata(
            model_name=str(config["model"]["name"]),
            class_names=tuple(str(name) for name in data["classes"]),
            crop_size=int(data["crop_size"]),
            mean=tuple(float(value) for value in data["input"]["mean"]),
            std=tuple(float(value) for value in data["input"]["std"]),
            label_version=str(state["label_version"]),
            step=int(state["step"]),
            best_score=float(state["best_score"]),
            sha256=_sha256(path),
        )
        weights = state["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{path} is not a NanoVision training checkpoint (the src/train.py format); "
            f"missing {error}"
        ) from error
    return weights, metadata


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
