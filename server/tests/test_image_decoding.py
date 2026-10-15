"""Decoding uploaded SEM images: TIFF is accepted, and 16-bit images keep every bit."""

import io

import numpy as np
import pytest
from PIL import Image

from app.api.image_utils import InvalidImageError, decode_image, validate_content_type


def tiff_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="TIFF")
    return buffer.getvalue()


def test_tiff_uploads_are_accepted() -> None:
    validate_content_type("image/tiff")  # does not raise


def test_8_bit_gray_tiff_becomes_rgb_with_unchanged_values() -> None:
    gray = np.random.default_rng(0).integers(0, 256, (20, 30), dtype=np.uint8)

    decoded = decode_image(tiff_bytes(gray))

    assert decoded.shape == (20, 30, 3) and decoded.dtype == np.uint8
    for channel in range(3):
        np.testing.assert_array_equal(decoded[:, :, channel], gray)


@pytest.mark.parametrize("byte_order", ["<u2", ">u2"])  # little- and big-endian 16-bit TIFFs
def test_16_bit_tiff_keeps_full_precision(byte_order: str) -> None:
    gray = np.random.default_rng(1).integers(0, 65536, (20, 30)).astype(byte_order)

    decoded = decode_image(tiff_bytes(gray))

    assert decoded.shape == (20, 30) and decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, gray)  # no value was cut to 8 bits


def test_32_bit_pixels_within_16_bit_range_are_accepted() -> None:
    pixels = np.array([[0, 1000], [40000, 65535]], dtype=np.int32)

    np.testing.assert_array_equal(decode_image(tiff_bytes(pixels)), pixels)


def test_32_bit_pixels_beyond_16_bits_are_rejected() -> None:
    pixels = np.array([[0, 70000]], dtype=np.int32)

    with pytest.raises(InvalidImageError, match="outside 0-65535"):
        decode_image(tiff_bytes(pixels))
