# Database session setup
# Creates the SQLite database and provides a session for each request

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# Path to the SQLite database file
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE_DIR, "..", "..", "student_predictions.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create the database engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory - creates one session per request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create the database tables if they don't exist yet"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Provide a database session to route functions (used with FastAPI Depends)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
