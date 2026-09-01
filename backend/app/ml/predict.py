# Prediction module
# Loads the trained model and makes a single prediction

import os
import json
import joblib
import numpy as np

# Path to the folder where trained models are saved
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "saved_models")

# Convert number labels back to text
LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Order of input features (must match training order)
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_gpa",
    "study_hours",
    "backlogs",
    "class_participation",
]


def predict(input_data):
    """
    Make a prediction for one student.

    input_data : a dictionary with all 7 features
    returns    : risk_level, confidence, and model name
    """
    # Step 1: Load the saved metrics to find out which model is best
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError("No trained models found. Run train.py first.")

    with open(metrics_path) as f:
        summary = json.load(f)

    best_model_name = summary["best_model"]

    # Step 2: Load the best model from disk
    filename  = best_model_name.lower().replace(" ", "_") + ".joblib"
    model_path = os.path.join(MODELS_DIR, filename)
    pipeline  = joblib.load(model_path)

    # Step 3: Build the input array in correct feature order
    import pandas as pd
    X = pd.DataFrame([[input_data[feature] for feature in FEATURES]], columns=FEATURES)

    # Step 4: Run the model to get a prediction
    prediction_number = pipeline.predict(X)[0]
    risk_level        = LABEL_MAP[int(prediction_number)]

    # Step 5: Get the confidence (probability of the predicted class)
    confidence = None
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        confidence    = round(float(max(probabilities)), 4)

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "model":      best_model_name,
    }
