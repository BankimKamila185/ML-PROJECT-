"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage (0–100)")
    internal_marks: float = Field(..., ge=0.0, le=100.0, description="Internal exam marks (0–100)")
    assignment_score: float = Field(..., ge=0.0, le=100.0, description="Assignment score (0–100)")
    previous_gpa: float = Field(..., ge=0.0, le=10.0, description="Previous semester GPA (0–10)")
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Daily study hours (0–24)")
    backlogs: int = Field(..., ge=0, description="Number of backlogs (0 or more)")
    class_participation: float = Field(..., ge=0.0, le=100.0, description="Class participation % (0–100)")


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    risk_level: str
    confidence: Optional[float] = None
    model: str
    recommendation: str


class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_labels: List[str]


class ModelsResponse(BaseModel):
    best_model: str
    models: dict


class HistoryRecord(BaseModel):
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
    confidence: Optional[float] = None
    model_used: str
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    total: int
    records: List[HistoryRecord]
