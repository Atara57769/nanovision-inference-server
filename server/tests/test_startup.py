from fastapi.testclient import TestClient

from app.main import create_app
from app.services.prediction_service import PredictionService


def test_lifespan_builds_prediction_service():
    app = create_app()
    with TestClient(app) as client:
        assert isinstance(client.app.state.prediction_service, PredictionService)


def test_state_cleared_after_shutdown():
    app = create_app()
    with TestClient(app):
        pass
    assert app.state.prediction_service is None
