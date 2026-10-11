from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    service_ready = getattr(request.app.state, "prediction_service", None) is not None
    return {"status": "ok", "service_ready": service_ready}
