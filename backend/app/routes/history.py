"""History route – GET /api/history and GET /api/dashboard"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from ..schemas import HistoryResponse
from ..services import db_service, ml_service
from ..database.session import get_db

router = APIRouter()


@router.get("/history", response_model=HistoryResponse, tags=["History"])
def get_history(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    risk_filter: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
):
    total, records = db_service.get_history(
        db,
        skip=skip,
        limit=limit,
        search=search,
        risk_filter=risk_filter,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return HistoryResponse(total=total, records=records)


@router.get("/dashboard", tags=["Dashboard"])
def get_dashboard(db: Session = Depends(get_db)):
    """Returns stats for dashboard KPI cards."""
    stats = db_service.get_summary_stats(db)

    model_info = {}
    if ml_service.is_trained():
        try:
            metrics = ml_service.get_metrics()
            best = metrics["best_model"]
            model_info = {
                "best_model": best,
                "best_model_accuracy": metrics["models"][best]["accuracy"],
                "best_model_f1": metrics["models"][best]["f1_score"],
            }
        except Exception:
            pass

    return {**stats, **model_info}
