import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api.prediction_controller import get_prediction_service
from app.main import create_app


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_ok():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service_ready": True}


def test_predict_valid_image_returns_mask():
    app = create_app()
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
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_predict_rejects_corrupt_image_bytes():
    app = create_app()
    with TestClient(app) as client:
        files = {"file": ("test.png", b"\x89PNG\r\n not really a png", "image/png")}
        response = client.post("/predict", files=files)
    assert response.status_code == 400


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
    with TestClient(app) as client:
        response = client.post("/predict")
    assert response.status_code == 422
