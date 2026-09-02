"""
Student Performance Risk Prediction - FastAPI Server
Module V: Supervised Learning - Classification

Consolidates:
  - FastAPI application and CORS middleware
  - Pydantic validation schemas
  - SQLite database ORM and operations
  - REST endpoints (/api/health, /api/predict, /api/models, /api/history, /api/dashboard)
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, Float, String, DateTime, func, create_engine, desc
from sqlalchemy.orm import declarative_base, sessionmaker, Session

import ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "student_predictions.db")

# ── Database Setup ────────────────────────────────────────────────────────────
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String)
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
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    student_id: Optional[str] = None
    attendance: float = Field(..., ge=0, le=100)
    internal_marks: float = Field(..., ge=0, le=100)
    assignment_score: float = Field(..., ge=0, le=100)
    previous_gpa: float = Field(..., ge=0, le=10)
    study_hours: float = Field(..., ge=0, le=24)
    backlogs: int = Field(..., ge=0)
    class_participation: float = Field(..., ge=0, le=100)
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    risk_level: str
    confidence: Optional[float]
    model: str
    recommendation: str


class HistoryRecordSchema(BaseModel):
    id: int
    student_id: str
    attendance: float
    internal_marks: float
    assignment_score: float
    previous_gpa: float
    study_hours: float
    backlogs: int
    class_participation: float
    risk_level: str
    confidence: Optional[float]
    model_used: str
    created_at: Optional[str]


class HistoryResponse(BaseModel):
    total: int
    records: List[HistoryRecordSchema]


# ── Lifespan & Application Setup ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ml.is_trained():
        print("Training models on initial launch...")
        ml.train_all_models()
    yield


app = FastAPI(
    title="Student Performance Risk Prediction API",
    description="Module V: Supervised Learning Classification API",
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


# ── Root Welcome / API Status ──────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return {
        "message": "Student Performance Risk Prediction API is running",
        "docs": "/docs",
        "frontend": "http://localhost:5173",
        "health": "/api/health"
    }


# ── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "trained": ml.is_trained(),
    }


@app.get("/api/models", tags=["ML"])
@app.get("/api/metrics", tags=["ML"])
def get_models():
    if not ml.is_trained():
        raise HTTPException(status_code=503, detail="Models not trained yet.")
    try:
        return ml.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse, tags=["ML"])
def predict(req: PredictionRequest, db: Session = Depends(get_db)):
    input_dict = req.model_dump()
    try:
        result = ml.predict_risk(input_dict, model_name=req.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Record prediction to SQLite
    try:
        student_id = req.student_id or ("STU-" + uuid.uuid4().hex[:8].upper())
        rec = PredictionRecord(
            student_id=student_id,
            attendance=req.attendance,
            internal_marks=req.internal_marks,
            assignment_score=req.assignment_score,
            previous_gpa=req.previous_gpa,
            study_hours=req.study_hours,
            backlogs=req.backlogs,
            class_participation=req.class_participation,
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            model_used=result["model"],
            created_at=datetime.now(timezone.utc),
        )
        db.add(rec)
        db.commit()
    except Exception as err:
        print(f"Failed to record history: {err}")

    return PredictionResponse(
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        model=result["model"],
        recommendation=result["recommendation"],
    )


@app.get("/api/history", response_model=HistoryResponse, tags=["History"])
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    risk_filter: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionRecord)
    if search:
        query = query.filter(
            PredictionRecord.student_id.ilike(f"%{search}%") |
            PredictionRecord.risk_level.ilike(f"%{search}%")
        )
    if risk_filter and risk_filter.upper() in ("LOW", "MEDIUM", "HIGH"):
        query = query.filter(PredictionRecord.risk_level == risk_filter.upper())

    sort_col = getattr(PredictionRecord, sort_by, PredictionRecord.created_at)
    query = query.order_by(sort_col.desc() if sort_order == "desc" else sort_col.asc())

    total = query.count()
    records = query.offset(skip).limit(limit).all()

    return HistoryResponse(
        total=total,
        records=[
            HistoryRecordSchema(
                id=r.id,
                student_id=r.student_id,
                attendance=r.attendance,
                internal_marks=r.internal_marks,
                assignment_score=r.assignment_score,
                previous_gpa=r.previous_gpa,
                study_hours=r.study_hours,
                backlogs=r.backlogs,
                class_participation=r.class_participation,
                risk_level=r.risk_level,
                confidence=r.confidence,
                model_used=r.model_used,
                created_at=r.created_at.isoformat() if r.created_at else None,
            )
            for r in records
        ],
    )


@app.get("/api/dashboard", tags=["Dashboard"])
def get_dashboard(db: Session = Depends(get_db)):
    total = db.query(func.count(PredictionRecord.id)).scalar() or 0
    low = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "LOW").scalar() or 0
    med = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "MEDIUM").scalar() or 0
    high = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "HIGH").scalar() or 0

    model_info = {}
    if ml.is_trained():
        try:
            metrics = ml.get_metrics()
            best = metrics["best_model"]
            model_info = {
                "best_model": best,
                "best_model_accuracy": metrics["models"][best]["accuracy"],
                "best_model_f1": metrics["models"][best]["f1_score"],
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
