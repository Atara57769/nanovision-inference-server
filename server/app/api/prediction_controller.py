from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File

from app.api.image_utils import InvalidImageError, decode_image, validate_image
from app.api.response_utils import serialize_result
from app.services.prediction_service import PredictionService

router = APIRouter()


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
) -> dict:
    content = await file.read()

    try:
        validate_image(file.content_type, content)
        image = decode_image(content)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = service.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return serialize_result(result)
