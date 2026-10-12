"""Split a whole image into overlapping tiles and merge tile predictions back.

Ported from NanoVision code/amat-2/src/predict.py (``tile_starts``,
``predict_probs``) @ d8f35ed on team/amat-2 -- keep in sync.

The model was trained on 256x256 native-resolution crops. Resizing a whole SEM
image to fit would change the particle scale the model learned, so the image is
cut into overlapping tiles instead, and the per-tile class probabilities are
averaged back into one full-size map.

NumPy only, on purpose: this module must not import torch or ``app.models``.
That keeps it testable without a checkpoint, and keeps the dependency one-way
(models use tiling; tiling knows nothing about models).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TilingMetadata:
    """Everything needed to rebuild a full-size map from per-tile outputs.

    Padding is only ever added at the bottom and right, so
    ``padded_shape - original_shape`` is the complete padding information.
    """

    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    tile_origins: tuple[tuple[int, int], ...]  # (y, x) of each tile, in tile order
    tile_size: int


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, length)`` with windows of ``tile_size``.

    The last window is moved back to end exactly at the edge, rather than
    sticking out past it, so every tile is full-size and real image content.
    """

    if tile_size <= 0 or stride <= 0:
        raise ValueError(f"tile_size and stride must be positive, got {tile_size} and {stride}")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def split_to_tiles(
    image: np.ndarray, tile_size: int, overlap: int
) -> tuple[np.ndarray, TilingMetadata]:
    """``[C, H, W]`` image -> ``[N, C, tile, tile]`` tiles plus their metadata.

    An image smaller than one tile is padded by repeating its edge pixels
    (research uses replicate padding). Edge padding commutes with per-channel
    normalisation, so padding before or after normalising gives the same tiles.
    """

    if image.ndim != 3:
        raise ValueError(f"expected a channel-first [C, H, W] image, got shape {image.shape}")
    channels, height, width = image.shape
    if channels == 0 or height == 0 or width == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < tile_size:
        # overlap == tile_size would mean a stride of 0: the tiles never advance.
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap} for tile_size {tile_size}")

    stride = tile_size - overlap
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    padded_h, padded_w = padded.shape[1:]

    # Row-major order (y outer, x inner), the same order research visits tiles.
    origins = tuple(
        (y, x)
        for y in tile_starts(padded_h, tile_size, stride)
        for x in tile_starts(padded_w, tile_size, stride)
    )
    tiles = np.stack([padded[:, y:y + tile_size, x:x + tile_size] for y, x in origins])
    metadata = TilingMetadata(
        original_shape=(height, width),
        padded_shape=(padded_h, padded_w),
        tile_origins=origins,
        tile_size=tile_size,
    )
    return tiles, metadata


def merge_predictions(predictions: np.ndarray, metadata: TilingMetadata) -> np.ndarray:
    """``[N, K, tile, tile]`` class probabilities -> ``[K, H_pad, W_pad]`` averages.

    Expects probabilities, not logits: averaging logits and then applying
    softmax is a different (and not the trained-and-evaluated) operation.
    """

    tile = metadata.tile_size
    if predictions.ndim != 4:
        raise ValueError(f"expected [N, K, tile, tile] predictions, got shape {predictions.shape}")
    if predictions.shape[0] != len(metadata.tile_origins):
        raise ValueError(
            f"got {predictions.shape[0]} tile predictions for {len(metadata.tile_origins)} tiles"
        )
    if predictions.shape[2:] != (tile, tile):
        # A model that returns a smaller map (e.g. SegFormer at 1/4 resolution)
        # must upsample inside its own wrapper; merging here would misplace pixels.
        raise ValueError(
            f"each tile prediction must be {tile}x{tile}, got {predictions.shape[2]}x{predictions.shape[3]}"
        )

    num_classes = predictions.shape[1]
    padded_h, padded_w = metadata.padded_shape
    # float32 and the same accumulation order as research, so merged values
    # match research's up to float rounding.
    sums = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
    counts = np.zeros((padded_h, padded_w), dtype=np.float32)
    for prediction, (y, x) in zip(predictions, metadata.tile_origins):
        sums[:, y:y + tile, x:x + tile] += prediction
        counts[y:y + tile, x:x + tile] += 1
    if np.any(counts == 0):
        raise ValueError("tile origins do not cover the padded image; metadata is inconsistent")
    return sums / counts


def remove_padding(array: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Crop ``[..., H_pad, W_pad]`` back to ``[..., H, W]`` (padding is bottom/right only)."""

    height, width = original_shape
    if array.ndim < 2 or height > array.shape[-2] or width > array.shape[-1]:
        raise ValueError(f"cannot crop array of shape {array.shape} to {original_shape}")
    return array[..., :height, :width]
