# Database model for storing prediction results
# Uses SQLAlchemy - a simple way to work with databases in Python

from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Base class that all database tables inherit from
Base = declarative_base()


class PredictionRecord(Base):
    # This is the name of the table in the database
    __tablename__ = "prediction_history"

    # Each variable below is one column in the table
    id               = Column(Integer, primary_key=True, autoincrement=True)
    student_id       = Column(String)
    attendance       = Column(Float)
    internal_marks   = Column(Float)
    assignment_score = Column(Float)
    previous_gpa     = Column(Float)
    study_hours      = Column(Float)
    backlogs         = Column(Integer)
    class_participation = Column(Float)
    risk_level       = Column(String)   # LOW / MEDIUM / HIGH
    confidence       = Column(Float)    # How confident the model is (0 to 1)
    model_used       = Column(String)   # Which ML model was used
    created_at       = Column(DateTime, default=datetime.utcnow)
