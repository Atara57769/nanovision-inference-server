from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.prediction_controller import get_prediction_service_or_none

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    service = get_prediction_service_or_none(request.app)
    service_ready = service is not None and service.is_ready
    return {"status": "ok", "service_ready": service_ready}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness probe: do not accept prediction traffic until weights are loaded."""

    service = get_prediction_service_or_none(request.app)
    service_ready = service is not None and service.is_ready
    return JSONResponse(
        status_code=200 if service_ready else 503,
        content={"status": "ready" if service_ready else "not_ready", "service_ready": service_ready},
    )
