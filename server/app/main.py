import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import prediction_controller, system_controller
from app.config.settings import SettingsError, load_settings
from app.inference.inference_pipeline import InferencePipeline
from app.model_management.model_provider import ModelLoadError, ModelProvider
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


def load_model_provider() -> ModelProvider | None:
    """Load the configured model once. If it cannot be loaded, the server still
    starts, as not ready: /health reports it and /predict answers 503."""
    try:
        provider = ModelProvider(load_settings())
        provider.load_active_model()
    except (OSError, SettingsError, ModelLoadError):
        logger.exception("The model could not be loaded; /predict answers 503 until this is fixed")
        return None
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = load_model_provider()
    pipeline = InferencePipeline(provider.get_model()) if provider is not None else None

    app.state.model_provider = provider
    app.state.prediction_service = PredictionService(pipeline)
    yield
    app.state.prediction_service = None
    app.state.model_provider = None


def create_app() -> FastAPI:
    app = FastAPI(title="NanoVision Inference Server", lifespan=lifespan)
    app.include_router(system_controller.router)
    app.include_router(prediction_controller.router)
    return app


app = create_app()
