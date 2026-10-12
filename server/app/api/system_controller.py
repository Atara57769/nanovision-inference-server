from fastapi import APIRouter, Request

from app.api.prediction_controller import get_prediction_service_or_none

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    service = get_prediction_service_or_none(request.app)
    service_ready = service is not None and service.is_ready
    return {"status": "ok", "service_ready": service_ready}
