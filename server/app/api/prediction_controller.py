from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.image_utils import (
    InvalidImageError,
    decode_image,
    read_upload_within_limit,
    validate_content_type,
    validate_image,
)
from app.api.response_utils import serialize_result
from app.services.prediction_service import PredictionService

router = APIRouter()


def get_prediction_service_or_none(app: FastAPI) -> PredictionService | None:
    return getattr(app.state, "prediction_service", None)


def get_prediction_service(request: Request) -> PredictionService:
    service = get_prediction_service_or_none(request.app)
    if service is None:
        raise HTTPException(status_code=503, detail="Prediction service not ready")
    return service


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    try:
        validate_content_type(file.content_type)  # cheap check first, before reading the body
        content = await read_upload_within_limit(file)
        validate_image(file.content_type, content)
        image = await run_in_threadpool(decode_image, content)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await run_in_threadpool(service.predict, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return serialize_result(result)
