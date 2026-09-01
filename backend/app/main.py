"""
FastAPI Application – Student Performance Risk Prediction
Module V: Supervised Learning – Classification
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database.session import init_db
from .routes import predict, models, history
from .services.ml_service import is_trained


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Student Performance Risk Prediction API",
    description=(
        "Module V – Supervised Learning: Classification. "
        "Predicts student academic risk using Logistic Regression, KNN, and Decision Tree."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the Vite dev server and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.get("/api/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "trained": is_trained(),
    }
