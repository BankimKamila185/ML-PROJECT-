# Prediction API route
# Handles POST /api/predict requests from the frontend

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..schemas          import PredictionRequest, PredictionResponse
from ..services         import ml_service, db_service
from ..database.session import get_db

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Accepts student data, runs the ML model, and returns the predicted risk level.
    Also saves the prediction to the database for history.
    """

    # Check if models have been trained
    if not ml_service.is_trained():
        raise HTTPException(
            status_code=503,
            detail="Models are not trained yet. Please run app/ml/train.py first."
        )

    # Run the prediction
    try:
        result = ml_service.run_prediction(request.model_dump())
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(error)}")

    # Save the result to the database (ignore errors so prediction still returns)
    try:
        db_service.create_record(
            db,
            input_data = request.model_dump(),
            risk_level = result["risk_level"],
            confidence = result["confidence"],
            model_used = result["model"],
        )
    except Exception:
        pass

    # Return the response
    return PredictionResponse(
        risk_level     = result["risk_level"],
        confidence     = result["confidence"],
        model          = result["model"],
        recommendation = result["recommendation"],
    )
