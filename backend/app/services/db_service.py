# Database service
# Functions to save and retrieve prediction records from the database

import uuid
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database.models import PredictionRecord


def create_record(db, input_data, risk_level, confidence, model_used):
    """
    Save one prediction result to the database.
    Generates a random student ID automatically.
    """
    # Generate a short random student ID like STU-AB12CD34
    student_id = "STU-" + uuid.uuid4().hex[:8].upper()

    # Create a new record object
    record = PredictionRecord(
        student_id          = student_id,
        attendance          = input_data["attendance"],
        internal_marks      = input_data["internal_marks"],
        assignment_score    = input_data["assignment_score"],
        previous_gpa        = input_data["previous_gpa"],
        study_hours         = input_data["study_hours"],
        backlogs            = input_data["backlogs"],
        class_participation = input_data["class_participation"],
        risk_level          = risk_level,
        confidence          = confidence,
        model_used          = model_used,
        created_at          = datetime.now(timezone.utc),
    )

    # Save to database
    db.add(record)
    db.commit()
    db.refresh(record)

    return record


def get_history(db, skip=0, limit=50, search=None, risk_filter=None,
                sort_by="created_at", sort_order="desc"):
    """
    Fetch prediction records from the database.
    Supports search, filtering by risk level, sorting, and pagination.
    """
    query = db.query(PredictionRecord)

    # Filter by search keyword (student ID or risk level)
    if search:
        query = query.filter(
            PredictionRecord.student_id.ilike(f"%{search}%") |
            PredictionRecord.risk_level.ilike(f"%{search}%")
        )

    # Filter by risk level
    if risk_filter and risk_filter.upper() in ("LOW", "MEDIUM", "HIGH"):
        query = query.filter(PredictionRecord.risk_level == risk_filter.upper())

    # Sort the results
    sort_column = getattr(PredictionRecord, sort_by, PredictionRecord.created_at)
    if sort_order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    total   = query.count()
    records = query.offset(skip).limit(limit).all()

    return total, records


def get_summary_stats(db):
    """Count how many students are LOW, MEDIUM, and HIGH risk"""
    total  = db.query(func.count(PredictionRecord.id)).scalar()
    low    = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "LOW").scalar()
    medium = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "MEDIUM").scalar()
    high   = db.query(func.count(PredictionRecord.id)).filter(PredictionRecord.risk_level == "HIGH").scalar()

    return {"total": total, "low": low, "medium": medium, "high": high}
