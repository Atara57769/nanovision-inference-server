"""Tests for the shared tiling utilities. No model or checkpoint is needed.

Several cases are ported from NanoVision code/amat-2/tests/test_three_class_tiling.py
(full coverage, small image padded and cropped back, wrong output size rejected).
"""

import numpy as np
import pytest

from app.inference.tiling import (
    TilingMetadata,
    merge_predictions,
    remove_padding,
    split_to_tiles,
    tile_starts,
)

TILE = 256
OVERLAP = 128


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (768, [0, 128, 256, 384, 512]),  # SEM image height: exact fit
        (1024, [0, 128, 256, 384, 512, 640, 768]),  # SEM image width: exact fit
        (700, [0, 128, 256, 384, 444]),  # last tile pulled back to end at the edge
        (256, [0]),  # exactly one tile
        (100, [0]),  # smaller than a tile: one (padded) tile
    ],
)
def test_tile_starts(length: int, expected: list[int]) -> None:
    assert tile_starts(length, TILE, TILE - OVERLAP) == expected


@pytest.mark.parametrize(("tile_size", "stride"), [(0, 128), (256, 0), (-1, 128)])
def test_tile_starts_rejects_non_positive_sizes(tile_size: int, stride: int) -> None:
    with pytest.raises(ValueError):
        tile_starts(768, tile_size, stride)


def test_full_sem_image_gives_35_tiles_in_row_major_order() -> None:
    image = np.random.default_rng(0).random((3, 768, 1024), dtype=np.float32)

    tiles, metadata = split_to_tiles(image, TILE, OVERLAP)

    assert tiles.shape == (35, 3, TILE, TILE)  # 5 rows x 7 columns
    assert metadata.original_shape == (768, 1024)
    assert metadata.padded_shape == (768, 1024)  # big enough: no padding
    assert metadata.tile_origins[:2] == ((0, 0), (0, 128))  # x varies fastest
    for tile, (y, x) in zip(tiles, metadata.tile_origins):
        np.testing.assert_array_equal(tile, image[:, y:y + TILE, x:x + TILE])


@pytest.mark.parametrize("shape", [(768, 1024), (700, 900), (256, 256), (257, 600), (100, 50)])
def test_split_then_merge_reconstructs_the_image(shape: tuple[int, int]) -> None:
    # An "identity model" returns its input, so a correct split+merge+crop must
    # give the original image back. This proves every pixel is covered and that
    # each tile is put back in the right place.
    image = np.random.default_rng(1).random((3, *shape), dtype=np.float32)

    tiles, metadata = split_to_tiles(image, TILE, OVERLAP)
    merged = remove_padding(merge_predictions(tiles, metadata), metadata.original_shape)

    assert merged.shape == image.shape
    np.testing.assert_allclose(merged, image, rtol=1e-6)


def test_small_image_is_padded_by_repeating_edges() -> None:
    image = np.arange(100 * 50, dtype=np.float32).reshape(1, 100, 50)

    tiles, metadata = split_to_tiles(image, TILE, OVERLAP)

    assert tiles.shape == (1, 1, TILE, TILE)
    assert metadata.original_shape == (100, 50)
    assert metadata.padded_shape == (TILE, TILE)
    padded = tiles[0, 0]
    np.testing.assert_array_equal(padded[:100, :50], image[0])
    # Padding copies the last real row/column, it does not insert zeros.
    np.testing.assert_array_equal(padded[150, :50], image[0, -1])
    np.testing.assert_array_equal(padded[:100, 200], image[0, :, -1])


def test_overlapping_tiles_are_averaged() -> None:
    # Width 384 -> two tiles starting at x=0 and x=128; they share columns 128-255.
    _, metadata = split_to_tiles(np.zeros((1, 256, 384), dtype=np.float32), TILE, OVERLAP)
    assert metadata.tile_origins == ((0, 0), (0, 128))
    left_tile_says = np.zeros((1, TILE, TILE), dtype=np.float32)
    right_tile_says = np.ones((1, TILE, TILE), dtype=np.float32)

    merged = merge_predictions(np.stack([left_tile_says, right_tile_says]), metadata)

    assert np.all(merged[0, :, :128] == 0.0)  # only the left tile
    assert np.all(merged[0, :, 128:256] == 0.5)  # both tiles: the mean
    assert np.all(merged[0, :, 256:] == 1.0)  # only the right tile


@pytest.mark.parametrize(
    ("image", "tile_size", "overlap"),
    [
        (np.zeros((768, 1024), dtype=np.float32), TILE, OVERLAP),  # missing channel axis
        (np.zeros((3, 0, 1024), dtype=np.float32), TILE, OVERLAP),  # empty image
        (np.zeros((3, 768, 1024), dtype=np.float32), TILE, TILE),  # stride would be 0
        (np.zeros((3, 768, 1024), dtype=np.float32), TILE, -1),  # negative overlap
        (np.zeros((3, 768, 1024), dtype=np.float32), 0, 0),  # no tile size
    ],
)
def test_split_rejects_invalid_input(image: np.ndarray, tile_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_to_tiles(image, tile_size, overlap)


def _two_tile_metadata() -> TilingMetadata:
    return TilingMetadata(
        original_shape=(256, 384),
        padded_shape=(256, 384),
        tile_origins=((0, 0), (0, 128)),
        tile_size=TILE,
    )


def test_merge_rejects_wrong_number_of_tiles() -> None:
    one_prediction = np.zeros((1, 3, TILE, TILE), dtype=np.float32)

    with pytest.raises(ValueError, match="tile predictions"):
        merge_predictions(one_prediction, _two_tile_metadata())


def test_merge_rejects_low_resolution_output() -> None:
    # What a model that forgets to upsample (SegFormer: 1/4 resolution) returns.
    quarter_size = np.zeros((2, 3, TILE // 4, TILE // 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must be 256x256"):
        merge_predictions(quarter_size, _two_tile_metadata())


def test_remove_padding_crops_bottom_and_right() -> None:
    padded = np.arange(3 * 256 * 256).reshape(3, 256, 256)

    cropped = remove_padding(padded, (100, 50))

    assert cropped.shape == (3, 100, 50)
    np.testing.assert_array_equal(cropped, padded[:, :100, :50])


def test_remove_padding_rejects_a_larger_target() -> None:
    with pytest.raises(ValueError):
        remove_padding(np.zeros((3, 100, 100)), (200, 100))
