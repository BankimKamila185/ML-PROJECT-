"""
Student Performance Risk Prediction - Machine Learning Engine
Module V: Supervised Learning - Classification

Consolidates:
  - Synthetic data generation (1,200 rows with realistic features & noise)
  - Data preprocessing (median imputation, feature scaling)
  - Model training (Logistic Regression, KNN, Decision Tree)
  - Model evaluation (Accuracy, Precision, Recall, F1 score, Confusion Matrix)
  - Automatic best model selection (highest F1 score)
  - Model persistence and inference execution
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_gpa",
    "study_hours",
    "backlogs",
    "class_participation",
]

LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REVERSE_LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
CLASSES = ["LOW", "MEDIUM", "HIGH"]


# ── Step 1: Data Generation ──────────────────────────────────────────────────
def generate_dataset(n=1200, seed=42):
    """Generates synthetic student performance data with realistic relationships."""
    np.random.seed(seed)
    attendance          = np.clip(np.random.normal(75, 15, n), 0, 100).round(1)
    internal_marks      = np.clip(np.random.normal(65, 18, n), 0, 100).round(1)
    assignment_score    = np.clip(np.random.normal(70, 15, n), 0, 100).round(1)
    previous_gpa        = np.clip(np.random.normal(7.0, 1.5, n), 0, 10).round(2)
    study_hours         = np.clip(np.random.normal(4, 2, n), 0, 14).round(1)
    backlogs            = np.random.choice([0, 0, 0, 1, 1, 2, 3, 4], size=n)
    class_participation = np.clip(np.random.normal(65, 20, n), 0, 100).round(1)

    score = (
        0.25 * attendance
        + 0.20 * internal_marks
        + 0.15 * assignment_score
        + 0.20 * (previous_gpa / 10 * 100)
        + 0.10 * (study_hours / 14 * 100)
        + 0.05 * class_participation
        - 5.0  * backlogs
    )
    score = np.clip(score + np.random.normal(0, 3, n), 0, 100)

    risk_level = [
        "LOW" if s > 70 else ("MEDIUM" if s > 50 else "HIGH") for s in score
    ]

    df = pd.DataFrame({
        "attendance":          attendance,
        "internal_marks":      internal_marks,
        "assignment_score":    assignment_score,
        "previous_gpa":        previous_gpa,
        "study_hours":         study_hours,
        "backlogs":            backlogs,
        "class_participation": class_participation,
        "risk_level":          risk_level,
    })

    rng = np.random.default_rng(seed)
    for col in ["attendance", "study_hours", "class_participation"]:
        df.loc[rng.random(n) < 0.02, col] = np.nan

    return df


# ── Step 2: Preprocessing Pipelines ──────────────────────────────────────────
def build_scaled_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_unscaled_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["risk_level"])
    X = df[FEATURES]
    y = df["risk_level"].map(LABEL_MAP).values
    return X, y


# ── Step 3: Evaluation Metrics ────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_labels": CLASSES,
    }


def select_best_model(all_metrics):
    return max(all_metrics, key=lambda m: (all_metrics[m]["f1_score"], all_metrics[m]["accuracy"]))


# ── Step 4: Training Runner ───────────────────────────────────────────────────
def train_all_models():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Generating dataset at {DATA_PATH}...")
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)

    X, y = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_scaled_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("preprocessor", build_scaled_pipeline()),
            ("classifier", KNeighborsClassifier(n_neighbors=7, metric="minkowski", p=2)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_unscaled_pipeline()),
            ("classifier", DecisionTreeClassifier(max_depth=10, min_samples_split=5, criterion="gini", random_state=42)),
        ]),
    }

    all_metrics = {}
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        all_metrics[model_name] = metrics

        filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(MODELS_DIR, filename))

    best_model_name = select_best_model(all_metrics)
    summary = {
        "best_model": best_model_name,
        "models": all_metrics,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed! Best model: {best_model_name}")
    return summary


# ── Step 5: Inference & Helper Functions ──────────────────────────────────────
def is_trained():
    return os.path.exists(METRICS_PATH)


def get_metrics():
    if not is_trained():
        raise FileNotFoundError("Models not trained yet.")
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict_risk(input_data: dict, model_name: str = None):
    if not is_trained():
        train_all_models()

    summary = get_metrics()
    selected_model = model_name or summary["best_model"]
    filename = selected_model.lower().replace(" ", "_") + ".joblib"
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {selected_model} not found.")

    pipeline = joblib.load(model_path)
    X = pd.DataFrame([[input_data[f] for f in FEATURES]], columns=FEATURES)

    pred_num = int(pipeline.predict(X)[0])
    risk_level = REVERSE_LABEL_MAP[pred_num]

    confidence = None
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = pipeline.predict_proba(X)[0]
        confidence = round(float(max(probabilities)), 4)

    # Intervention recommendation
    recs = []
    if risk_level == "HIGH":
        recs.append("CRITICAL: Schedule immediate 1-on-1 counseling with academic advisor.")
    elif risk_level == "MEDIUM":
        recs.append("MODERATE: Recommend weekly progress tracking and supplemental peer tutoring.")
    else:
        recs.append("GOOD STANDING: Student is demonstrating strong performance. Encourage honors/mentorship.")

    if input_data.get("attendance", 100) < 75:
        recs.append(f"Attendance ({input_data['attendance']}%) is below standard 75% threshold.")
    if input_data.get("backlogs", 0) > 0:
        recs.append(f"Enroll student into remedial sessions for {input_data['backlogs']} pending backlog(s).")
    if input_data.get("internal_marks", 100) < 50:
        recs.append("Internal marks require targeted concept reinforcement.")

    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "model": selected_model,
        "recommendation": " | ".join(recs),
    }


if __name__ == "__main__":
    train_all_models()
