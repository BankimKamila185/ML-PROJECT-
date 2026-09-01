# Model metrics route
# Handles GET /api/models - returns performance of all 3 trained models

from fastapi import APIRouter, HTTPException
from ..services import ml_service

router = APIRouter()


@router.get("/models")
def get_models():
    """Returns accuracy, precision, recall, F1 score, and confusion matrix for all 3 models."""

    if not ml_service.is_trained():
        raise HTTPException(status_code=503, detail="Models not trained yet.")

    try:
        data = ml_service.get_metrics()
        return data
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/metrics")
def get_metrics():
    """Same as /api/models - returns detailed metrics including confusion matrices."""
    return get_models()
