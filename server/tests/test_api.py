import asyncio
import io

import numpy as np
import PIL.Image
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.image_utils import MAX_FILE_SIZE_BYTES, InvalidImageError, read_upload_within_limit
from app.api.prediction_controller import get_prediction_service
from app.main import create_app
from app.models.prediction_result import PredictionResult
from app.services.prediction_service import PredictionService


class _StubReadyService:
    """A ready (is_ready=True) stand-in for PredictionService, for tests that
    exercise validation/decoding and don't care about prediction itself."""

    is_ready = True

    def predict(self, image: np.ndarray) -> PredictionResult:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        return PredictionResult(mask=mask, model_name="test-stub", model_version="v0", inference_time_ms=0.0)


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_large_png_bytes(size: int = 1000) -> bytes:
    """~1-3MB of incompressible (random) pixel data — big enough that a 1MB
    part-size cap would reject it, small enough to stay under the 10MB limit."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_ok_but_not_ready_with_stub_pipeline():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # A model that failed to load leaves the application alive but not ready.
    assert body["service_ready"] is False


def test_health_reports_ready_when_pipeline_is_loaded():
    app = create_app()
    with TestClient(app) as client:
        app.state.prediction_service = PredictionService(pipeline=object())
        response = client.get("/health")
    assert response.json()["service_ready"] is True


def test_predict_valid_image_returns_mask():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.png", _make_png_bytes(), "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 200
    body = response.json()
    assert "mask" in body
    assert "model_name" in body
    assert "model_version" in body
    assert "inference_time_ms" in body


def test_predict_rejects_invalid_content_type():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_accepts_image_jpg_content_type():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.jpg", _make_png_bytes(), "image/jpg")}
        response = client.post("/predict", files=files)
    assert response.status_code == 200


def test_predict_accepts_file_between_1mb_and_10mb():
    content = _make_large_png_bytes()
    assert 1024 * 1024 < len(content) < MAX_FILE_SIZE_BYTES, "fixture must land strictly between 1MB and 10MB"
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("big.png", content, "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 200


def test_predict_rejects_wrong_content_type_for_large_file():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    large_wrong_type_content = b"x" * (3 * 1024 * 1024)
    with TestClient(app) as client:
        files = {"file": ("big.txt", large_wrong_type_content, "text/plain")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "content type" in response.json()["detail"].lower()


def test_predict_rejects_wrong_content_type_without_reading_body(monkeypatch):
    import app.api.prediction_controller as controller_module

    was_called = False

    async def _fake_read_upload_within_limit(file):
        nonlocal was_called
        was_called = True
        return b""

    monkeypatch.setattr(controller_module, "read_upload_within_limit", _fake_read_upload_within_limit)

    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert was_called is False  # content-type check short-circuited before the heavy read


def test_predict_rejects_corrupt_image_bytes():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.png", b"\x89PNG\r\n not really a png", "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400
    # the client gets a generic message, not the raw PIL/internal exception text
    assert response.json()["detail"] == "Could not decode image"


def test_predict_maps_pipeline_failure_to_500():
    app = create_app()

    class FailingService:
        def predict(self, image):
            raise RuntimeError("boom")

    with TestClient(app) as client:
        app.dependency_overrides[get_prediction_service] = lambda: FailingService()
        files = {"file": ("test.png", _make_png_bytes(), "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 500


def test_predict_requires_file():
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        response = client.post("/predict")
    assert response.status_code == 422


def test_predict_rejects_decompression_bomb(monkeypatch):
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 10)
    app = create_app()
    app.dependency_overrides[get_prediction_service] = lambda: _StubReadyService()
    with TestClient(app) as client:
        files = {"file": ("test.png", _make_png_bytes(), "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_returns_503_when_pipeline_not_ready():
    """A failed or unavailable model must produce 503, not a 500/crash."""
    app = create_app()
    with TestClient(app) as client:
        files = {"file": ("test.png", _make_png_bytes(), "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 503


class _FakeUploadFile:
    """Stands in for FastAPI's UploadFile: an endless byte stream read in chunks."""

    def __init__(self, total_size: int):
        self._remaining = total_size

    async def read(self, size: int) -> bytes:
        if self._remaining <= 0:
            return b""
        chunk_size = min(size, self._remaining)
        self._remaining -= chunk_size
        return b"x" * chunk_size


def test_read_upload_within_limit_aborts_before_consuming_whole_stream():
    huge_size = MAX_FILE_SIZE_BYTES * 10
    fake_file = _FakeUploadFile(total_size=huge_size)

    with pytest.raises(InvalidImageError):
        asyncio.run(read_upload_within_limit(fake_file))

    assert fake_file._remaining > 0  # aborted long before consuming the whole fake stream
