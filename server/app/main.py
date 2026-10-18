import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import prediction_controller, system_controller
from app.api.image_utils import MAX_FILE_SIZE_BYTES
from app.api.request_size_limit import MaxBodySizeMiddleware
from app.config.settings import SettingsError, load_settings
from app.inference.inference_pipeline import InferencePipeline
from app.model_management.model_provider import ModelLoadError, ModelProvider
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

# Multipart framing (boundaries, part headers) adds a little overhead on top of
# the file content itself; MAX_FILE_SIZE_BYTES alone would reject otherwise-valid
# uploads right at the limit.
MAX_REQUEST_BODY_BYTES = MAX_FILE_SIZE_BYTES + 64 * 1024

# Comma-separated browser origins allowed to call this API. Defaults to the
# Vite dev server so local development keeps working with no setup; any other
# deployment (staging, production, a different dev port) must set this itself.
ALLOWED_ORIGINS_ENV = "NANOVISION_ALLOWED_ORIGINS"
_DEFAULT_ALLOWED_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def allowed_origins() -> list[str]:
    raw = os.environ.get(ALLOWED_ORIGINS_ENV)
    if not raw:
        return list(_DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


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
    # Caps the request body before Starlette's multipart parser can buffer an
    # oversized upload to memory/disk.
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)
    # The client is a browser app on its own origin (Vite dev server by
    # default); without this, every fetch() from it fails CORS before the
    # request is even sent. See ALLOWED_ORIGINS_ENV to deploy elsewhere.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(system_controller.router)
    app.include_router(prediction_controller.router)
    return app


app = create_app()
