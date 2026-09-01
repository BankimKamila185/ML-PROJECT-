# ML Service
# Loads trained models and runs predictions
# Models are loaded once and kept in memory (so we don't reload from disk every time)

import os
import json
import joblib
import numpy as np

# Path to the folder where trained models are saved
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "saved_models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Labels: model outputs 0/1/2, we convert back to text
LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Feature names in the correct order
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_gpa",
    "study_hours",
    "backlogs",
    "class_participation",
]

# Rule-based recommendations for each risk level
RECOMMENDATIONS = {
    "LOW":    "Student performance appears stable. Continue maintaining current academic habits.",
    "MEDIUM": "Focus on improving attendance, assignments, and study consistency.",
    "HIGH":   "Student may require additional academic support. Focus on attendance, internal marks, and regular study.",
}

# Cache so we load models only once
_loaded_metrics   = None
_loaded_pipelines = {}


def is_trained():
    """Returns True if models have been trained (metrics.json exists)"""
    return os.path.exists(METRICS_PATH)


def get_metrics():
    """Load and return the saved metrics from metrics.json"""
    global _loaded_metrics

    if _loaded_metrics is None:
        if not os.path.exists(METRICS_PATH):
            raise FileNotFoundError("Models not trained. Run train.py first.")
        with open(METRICS_PATH) as f:
            _loaded_metrics = json.load(f)

    return _loaded_metrics


def get_pipeline(model_name):
    """Load a trained pipeline from disk (cached after first load)"""
    global _loaded_pipelines

    if model_name not in _loaded_pipelines:
        filename   = model_name.lower().replace(" ", "_") + ".joblib"
        model_path = os.path.join(MODELS_DIR, filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _loaded_pipelines[model_name] = joblib.load(model_path)

    return _loaded_pipelines[model_name]


def run_prediction(input_data):
    """
    Run a prediction using the best trained model.

    input_data : dictionary with 7 feature values
    returns    : risk level, confidence, model name, and recommendation
    """
    # Find out which model is best
    summary         = get_metrics()
    best_model_name = summary["best_model"]
    pipeline        = get_pipeline(best_model_name)

    # Build input array (as DataFrame so sklearn knows feature names)
    import pandas as pd
    X = pd.DataFrame([[input_data[f] for f in FEATURES]], columns=FEATURES)

    # Predict
    prediction_number = pipeline.predict(X)[0]
    risk_level        = LABEL_MAP[int(prediction_number)]

    # Confidence
    confidence = None
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        confidence    = round(float(max(probabilities)), 4)

    return {
        "risk_level":     risk_level,
        "confidence":     confidence,
        "model":          best_model_name,
        "recommendation": RECOMMENDATIONS[risk_level],
    }
