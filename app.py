"""
Student Performance Risk Prediction - Web API & Full-Stack Server
Module V: Supervised Learning - Classification

Features:
  - Serves REST API for ML predictions, metrics, and history
  - Serves single-file modern dashboard UI at http://localhost:8000
  - Manages SQLite prediction history database
  - Automatically triggers training on startup if models are missing
"""

import os
import json
from datetime import datetime
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, Float, String, DateTime, create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Import ML training routine from our train.py
from train import FEATURES, LABEL_MAP, REVERSE_LABEL_MAP, train_all_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DB_PATH = os.path.join(BASE_DIR, "student_predictions.db")
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

# ── Database Setup ────────────────────────────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, default="STU-ANON")
    attendance = Column(Float)
    internal_marks = Column(Float)
    assignment_score = Column(Float)
    previous_gpa = Column(Float)
    study_hours = Column(Float)
    backlogs = Column(Integer)
    class_participation = Column(Float)
    risk_level = Column(String)
    confidence = Column(Float)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    student_id: Optional[str] = "STU-001"
    attendance: float = Field(..., ge=0, le=100)
    internal_marks: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    previous_gpa: float = Field(..., ge=0, le=10)
    study_hours: float = Field(..., ge=0, le=14)
    backlogs: int = Field(..., ge=0)
    class_participation: float = Field(..., ge=0, le=100)
    model_name: Optional[str] = None  # None selects best model


class PredictionResponse(BaseModel):
    risk_level: str
    confidence: float
    model: str
    probabilities: dict
    recommendation: str


from contextlib import asynccontextmanager

def ensure_models_exist():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print("Trained models not detected. Running initial model training...")
        train_all_models(BASE_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_models_exist()
    yield

# ── FastAPI App Initialization ────────────────────────────────────────────────
app = FastAPI(
    title="Student Performance Risk Prediction",
    description="Full-stack Supervised Learning classification system.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend HTML Delivery ────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML)
    return {"message": "index.html not found. Place index.html in the project root."}


# ── Prediction Engine & Recommendations ───────────────────────────────────────
def generate_recommendations(features: dict, risk_level: str) -> str:
    recs = []
    if risk_level == "HIGH":
        recs.append("CRITICAL: Schedule immediate 1-on-1 counseling with academic advisor.")
    elif risk_level == "MEDIUM":
        recs.append("MODERATE: Recommend weekly progress tracking and supplemental peer tutoring.")
    else:
        recs.append("GOOD STANDING: Student is demonstrating strong performance. Encourage honors/mentorship.")

    if features["attendance"] < 75:
        recs.append(f"Attendance ({features['attendance']}%) is below standard 75% threshold.")
    if features["backlogs"] > 0:
        recs.append(f"Enroll student into remedial sessions for {features['backlogs']} pending backlog(s).")
    if features["internal_marks"] < 50:
        recs.append("Internal marks require targeted concept reinforcement.")
    if features["study_hours"] < 2.5:
        recs.append("Recommend structured study schedules to increase daily independent study hours.")

    return " | ".join(recs)


# ── REST API Endpoints ────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    return {
        "status": "ok",
        "trained": os.path.exists(metrics_path),
    }


@app.get("/api/models", tags=["ML"])
@app.get("/api/metrics", tags=["ML"])
def get_metrics():
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=503, detail="Models are not trained yet.")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/api/predict", response_model=PredictionResponse, tags=["ML"])
def predict_risk(request: PredictionRequest, db: Session = Depends(get_db)):
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=503, detail="Models are not trained yet.")

    with open(metrics_path) as f:
        summary = json.load(f)

    # Use requested model or fall back to automatically selected best model
    model_name = request.model_name or summary["best_model"]
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")

    pipeline = joblib.load(model_path)

    # Build feature DataFrame
    input_dict = request.model_dump()
    X = pd.DataFrame([[input_dict[col] for col in FEATURES]], columns=FEATURES)

    # Inference
    pred_idx = int(pipeline.predict(X)[0])
    risk_level = REVERSE_LABEL_MAP.get(pred_idx, "UNKNOWN")

    # Probabilities
    probs = {"LOW": 0.0, "MEDIUM": 0.0, "HIGH": 0.0}
    confidence = 1.0

    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        raw_probs = pipeline.predict_proba(X)[0]
        for idx, prob in enumerate(raw_probs):
            name = REVERSE_LABEL_MAP.get(idx, f"CLASS_{idx}")
            probs[name] = round(float(prob), 4)
        confidence = round(float(max(raw_probs)), 4)

    recommendation = generate_recommendations(input_dict, risk_level)

    # Record to DB
    try:
        record = PredictionRecord(
            student_id=request.student_id or "STU-ANON",
            attendance=request.attendance,
            internal_marks=request.internal_marks,
            assignment_score=request.assignment_score,
            previous_gpa=request.previous_gpa,
            study_hours=request.study_hours,
            backlogs=request.backlogs,
            class_participation=request.class_participation,
            risk_level=risk_level,
            confidence=confidence,
            model_used=model_name,
        )
        db.add(record)
        db.commit()
    except Exception as e:
        print(f"Failed to log record to DB: {e}")

    return PredictionResponse(
        risk_level=risk_level,
        confidence=confidence,
        model=model_name,
        probabilities=probs,
        recommendation=recommendation,
    )


@app.get("/api/history", tags=["History"])
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    risk_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionRecord)
    if risk_filter and risk_filter.upper() in ["LOW", "MEDIUM", "HIGH"]:
        query = query.filter(PredictionRecord.risk_level == risk_filter.upper())
    if search:
        query = query.filter(PredictionRecord.student_id.ilike(f"%{search}%"))

    total = query.count()
    records = query.order_by(desc(PredictionRecord.created_at)).offset(skip).limit(limit).all()

    return {
        "total": total,
        "records": [
            {
                "id": r.id,
                "student_id": r.student_id,
                "attendance": r.attendance,
                "internal_marks": r.internal_marks,
                "assignment_score": r.assignment_score,
                "previous_gpa": r.previous_gpa,
                "study_hours": r.study_hours,
                "backlogs": r.backlogs,
                "class_participation": r.class_participation,
                "risk_level": r.risk_level,
                "confidence": r.confidence,
                "model_used": r.model_used,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }


@app.get("/api/dashboard", tags=["Dashboard"])
def get_dashboard(db: Session = Depends(get_db)):
    total = db.query(PredictionRecord).count()
    low = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "LOW").count()
    med = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "MEDIUM").count()
    high = db.query(PredictionRecord).filter(PredictionRecord.risk_level == "HIGH").count()

    model_info = {}
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            best = data["best_model"]
            model_info = {
                "best_model": best,
                "best_model_accuracy": data["models"][best]["accuracy"],
                "best_model_f1": data["models"][best]["f1_score"],
            }
        except Exception:
            pass

    return {
        "total": total,
        "low": low,
        "medium": med,
        "high": high,
        **model_info,
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Student Risk Prediction Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
