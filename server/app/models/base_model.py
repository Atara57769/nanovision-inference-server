"""The contract every segmentation model in the server implements.

Per request, ``InferencePipeline`` (app/inference/inference_pipeline.py) calls

    prepared = model.preprocess(image)
    predictions = [model.predict(batch) for batch in prepared.batches]
    mask = model.postprocess(predictions, prepared.metadata)

and at startup the model provider calls ``load()`` exactly once.

Template Method: everything the research-trained models share -- reading the
image the way training did, tiling, batching, merging, argmax and loading a
training checkpoint -- is written once, here. A concrete model only says which
network it is (``name`` and ``_build_network``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn

from app.inference.tiling import TilingMetadata, merge_predictions, remove_padding, split_to_tiles
from app.models.checkpoint import CheckpointMetadata, load_checkpoint
from app.models.classes import CLASS_NAMES, NUM_CLASSES, SegmentationClass

# One grayscale image repeated three times, because the encoders were pretrained
# on 3-channel ImageNet photos. These are not three different measurements.
INPUT_CHANNELS = 3

_UINT8_MAX = 255.0
_UINT16_MAX = 65535.0


@dataclass(frozen=True)
class ModelInfo:
    """Which model answered; returned to the client with every prediction."""

    name: str
    version: str


@dataclass(frozen=True)
class ReconstructionMetadata:
    """What ``postprocess`` needs to turn tile predictions back into one mask."""

    image_shape: tuple[int, int]  # (H, W) of the request image = shape of the returned mask
    tiling: TilingMetadata  # tiles of the predicted region: the image without its footer


@dataclass(frozen=True)
class PreprocessResult:
    """The model input for one image, in the shape ``InferencePipeline`` expects."""

    batches: list[torch.Tensor]  # each [b, 3, tile, tile] with b <= batch_size, in tile order
    metadata: ReconstructionMetadata


@dataclass(frozen=True)
class _LoadedState:
    """Everything ``load()`` produces, set in one step so a failed load leaves nothing half-done."""

    network: nn.Module
    device: torch.device
    checkpoint: CheckpointMetadata


def to_gray01(image: np.ndarray) -> np.ndarray:
    """Decoded image -> ``[H, W]`` float32 grayscale in [0, 1], exactly as training read images.

    Ported from NanoVision datasets/sem/image_utils.py ``load_grayscale01`` @ d8f35ed.
    8-bit colour input goes through PIL's ``convert("L")``, the call training used:
    for a gray image stored as RGB (R = G = B) it returns the gray value unchanged.
    16-bit input is divided by 65535 instead, because ``convert("L")`` would cut it
    to 8 bits. The division is done in float64 and then cast, like training did.
    """

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"expected a non-empty [H, W] or [H, W, C] image, got shape {image.shape}")

    if image.dtype == np.uint16 and image.ndim == 2:
        return (image.astype(np.float64) / _UINT16_MAX).astype(np.float32)
    if image.dtype == np.uint8 and image.ndim == 2:
        gray = image
    elif image.dtype == np.uint8 and image.shape[2] in (3, 4):  # RGB or RGBA
        gray = np.asarray(Image.fromarray(np.ascontiguousarray(image)).convert("L"))
    else:
        # Guessing the scale of any other format could silently shift every pixel.
        raise ValueError(
            "expected a uint8 gray/RGB/RGBA image or a uint16 gray image, "
            f"got dtype {image.dtype} with shape {image.shape}"
        )
    return (gray.astype(np.float64) / _UINT8_MAX).astype(np.float32)


def to_model_input(gray01: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """``[H, W]`` in [0, 1] -> ``[3, H, W]`` float32, normalised exactly once.

    Ported from NanoVision code/amat-2/src/data.py ``to_model_input`` @ d8f35ed.
    """

    stacked = np.repeat(gray01[None].astype(np.float32), INPUT_CHANNELS, axis=0)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(INPUT_CHANNELS, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(INPUT_CHANNELS, 1, 1)
    return (stacked - mean_array) / std_array


class BaseModel(ABC):
    """A segmentation model: whole image in, ``[H, W]`` mask of ``SegmentationClass`` ids out."""

    # Each concrete model sets its ModelName; it must equal model.name in its checkpoint.
    name: ClassVar[str]

    def __init__(self, *, tile_size: int, overlap: int, batch_size: int, footer_height: int = 0) -> None:
        """
        ``footer_height`` is the number of bottom rows holding the microscope's
        info bar (text and scale bar). Those rows are not predicted and come back
        as background. 0 means the images have no footer.
        """

        # Checked here, not per request, so a bad configuration stops the server at startup.
        if tile_size <= 0 or not 0 <= overlap < tile_size:
            raise ValueError(f"need tile_size > 0 and 0 <= overlap < tile_size, got {tile_size} and {overlap}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if footer_height < 0:
            raise ValueError(f"footer_height must not be negative, got {footer_height}")
        self._tile_size = tile_size
        self._overlap = overlap
        self._batch_size = batch_size
        self._footer_height = footer_height
        self._loaded: _LoadedState | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @abstractmethod
    def _build_network(self) -> nn.Module:
        """A new network with untrained weights; ``load()`` puts the trained ones in."""

    # 1. Lifecycle: called once, at startup.

    def load(self, checkpoint_path: Path, device: torch.device) -> None:
        """Build the network, fill in the trained weights and prepare it for inference."""

        weights, checkpoint = load_checkpoint(checkpoint_path)
        self._check_compatible(checkpoint_path, checkpoint)
        network = self._build_network()
        try:
            network.load_state_dict(weights)  # strict: every weight must match exactly
        except RuntimeError as error:
            raise ValueError(
                f"the weights in {checkpoint_path} do not fit the '{self.name}' network: {error}"
            ) from error
        # eval(): BatchNorm uses its learned statistics instead of the current
        # batch's, so a tile's result does not depend on which tiles share its batch.
        # requires_grad_(False): the server never trains, so no gradients are tracked.
        network.to(device).eval().requires_grad_(False)
        self._loaded = _LoadedState(network=network, device=device, checkpoint=checkpoint)

    # 2. Request path: called by InferencePipeline, in this order.

    def preprocess(self, image: np.ndarray) -> PreprocessResult:
        """Whole decoded image -> batches of normalised tiles, plus how to put them back.

        The image is never resized: particles must keep the scale the model was
        trained on. Large images are tiled, small ones padded.
        """

        loaded = self._require_loaded()  # the normalisation comes from the checkpoint
        gray = to_gray01(image)
        image_shape = (gray.shape[0], gray.shape[1])
        predicted_rows = image_shape[0] - self._footer_height
        if predicted_rows <= 0:
            raise ValueError(
                f"the image has {image_shape[0]} rows, not more than footer_height={self._footer_height}"
            )
        model_input = to_model_input(gray[:predicted_rows], loaded.checkpoint.mean, loaded.checkpoint.std)
        tiles, tiling = split_to_tiles(model_input, self._tile_size, self._overlap)
        batches = list(torch.from_numpy(tiles).split(self._batch_size))
        return PreprocessResult(
            batches=batches,
            metadata=ReconstructionMetadata(image_shape=image_shape, tiling=tiling),
        )

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """``[b, 3, tile, tile]`` -> raw logits ``[b, NUM_CLASSES, tile, tile]``.

        Output channel k is ``SegmentationClass(k)``. No softmax here: it is
        applied once, in ``postprocess``, before the tiles are averaged.
        """

        loaded = self._require_loaded()
        return loaded.network(batch.to(loaded.device))

    def postprocess(self, predictions: list[torch.Tensor], metadata: ReconstructionMetadata) -> np.ndarray:
        """Per-batch logits -> ``[H, W]`` uint8 mask of ``SegmentationClass`` ids.

        Mirrors research ``predict_probs``: softmax per tile, average the
        probabilities where tiles overlap, crop the padding, then argmax.
        """

        if not predictions:
            raise ValueError("postprocess needs at least one batch of predictions")
        logits = torch.cat(predictions)  # one [N, K, tile, tile] block, still in tile order
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"expected {NUM_CLASSES} class channels, got {logits.shape[1]}")
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        merged = merge_predictions(probabilities, metadata.tiling)
        region = remove_padding(merged, metadata.tiling.original_shape)
        labels = region.argmax(axis=0).astype(np.uint8)

        # Footer rows were never predicted; they hold no particles.
        mask = np.full(metadata.image_shape, SegmentationClass.BACKGROUND, dtype=np.uint8)
        mask[: labels.shape[0]] = labels
        return mask

    def info(self) -> ModelInfo:
        loaded = self._require_loaded()
        return ModelInfo(name=self.name, version=loaded.checkpoint.version)

    # Helpers

    def _check_compatible(self, checkpoint_path: Path, checkpoint: CheckpointMetadata) -> None:
        """Refuse a checkpoint the server would otherwise misuse without any error."""

        problems = []
        if checkpoint.model_name != self.name:
            problems.append(f"it holds a '{checkpoint.model_name}' model, not '{self.name}'")
        if checkpoint.class_names != CLASS_NAMES:
            # The dangerous one: wrong order would swap dark and bright in every mask.
            problems.append(f"it was trained with classes {checkpoint.class_names}, the server uses {CLASS_NAMES}")
        if checkpoint.crop_size != self._tile_size:
            problems.append(f"it was trained on {checkpoint.crop_size}px crops, but tile_size is {self._tile_size}")
        if len(checkpoint.mean) != INPUT_CHANNELS or len(checkpoint.std) != INPUT_CHANNELS:
            problems.append(f"its normalisation does not have {INPUT_CHANNELS} channels")
        if problems:
            raise ValueError(f"cannot use checkpoint {checkpoint_path}: " + "; ".join(problems))

    def _require_loaded(self) -> _LoadedState:
        if self._loaded is None:
            raise RuntimeError(f"the '{self.name}' model is not loaded; load() must run once at startup")
        return self._loaded
