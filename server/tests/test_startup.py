from fastapi.testclient import TestClient

from app.main import create_app
from app.services.prediction_service import PredictionService


def test_lifespan_builds_prediction_service():
    app = create_app()
    with TestClient(app) as client:
        assert isinstance(client.app.state.prediction_service, PredictionService)


def test_service_exists_but_not_ready_with_stub_pipeline():
    """The service object is always built at startup, even with the placeholder
    (None) pipeline — is_ready must distinguish "built" from "actually ready"."""
    app = create_app()
    with TestClient(app) as client:
        service = client.app.state.prediction_service
        assert service is not None
        assert service.is_ready is False


def test_state_cleared_after_shutdown():
    app = create_app()
    with TestClient(app):
        pass
    assert app.state.prediction_service is None
