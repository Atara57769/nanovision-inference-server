from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import prediction_controller, system_controller
from app.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO(part 3): replace with the real model loader.
    model = None
    # TODO(part 2): replace with the real InferencePipeline(model).
    pipeline = model

    app.state.prediction_service = PredictionService(pipeline)
    yield
    app.state.prediction_service = None


def create_app() -> FastAPI:
    app = FastAPI(title="NanoVision Inference Server", lifespan=lifespan)
    app.include_router(system_controller.router)
    app.include_router(prediction_controller.router)
    return app


app = create_app()
