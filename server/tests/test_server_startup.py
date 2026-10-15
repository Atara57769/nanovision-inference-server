"""The whole server: startup loads the configured model once, and /predict answers with it."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config.settings import CHECKPOINT_PATH_ENV, load_settings
from app.main import create_app
from tests.helpers import research_unetpp


def tiff_bytes(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="TIFF")
    return buffer.getvalue()


@pytest.fixture
def ready_app(write_checkpoint, monkeypatch: pytest.MonkeyPatch):
    # Untrained weights: enough to prove the wiring; accuracy is checked against research separately.
    checkpoint = write_checkpoint(research_unetpp().state_dict(), model_name="unetpp")
    monkeypatch.setenv(CHECKPOINT_PATH_ENV, str(checkpoint))
    return create_app()


def test_startup_loads_the_model_once_and_the_server_is_ready(ready_app) -> None:
    with TestClient(ready_app) as client:
        assert client.get("/health").json()["service_ready"] is True
        assert ready_app.state.model_provider.get_model().is_loaded


@pytest.mark.parametrize(
    "pixels",
    [
        pytest.param(np.random.default_rng(0).integers(0, 256, (300, 400), dtype=np.uint8), id="8-bit"),
        pytest.param(np.random.default_rng(0).integers(0, 65536, (300, 400)).astype(np.uint16), id="16-bit"),
    ],
)
def test_predict_answers_a_sem_tiff_with_a_full_size_mask(ready_app, pixels: np.ndarray) -> None:
    with TestClient(ready_app) as client:
        response = client.post("/predict", files={"file": ("sem.tif", tiff_bytes(pixels), "image/tiff")})

    assert response.status_code == 200
    body = response.json()
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["mask"]))))
    assert mask.shape == (300, 400)
    assert set(np.unique(mask)) <= {0, 1, 2}
    footer_height = load_settings().footer_height  # whatever the shipped config says
    assert np.all(mask[mask.shape[0] - footer_height :] == 0)  # the info bar comes back as background
    assert body["model_name"] == "unetpp"
    assert body["model_version"].startswith("v1-step100-")


def test_a_model_that_cannot_load_leaves_the_server_up_but_not_ready() -> None:
    # The autouse fixture points the checkpoint at a file that does not exist.
    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health").json()
        response = client.post("/predict", files={"file": ("sem.tif", tiff_bytes(np.zeros((8, 8), np.uint8)), "image/tiff")})

    assert health["service_ready"] is False
    assert response.status_code == 503
    assert app.state.model_provider is None
