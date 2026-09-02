"""
Student Performance Risk Prediction - Training & ML Pipeline
Module V: Supervised Learning - Classification

This script handles:
  1. Synthetic data generation (1,200 rows with realistic features & noise)
  2. Data preprocessing (median imputation, feature scaling)
  3. Model training (Logistic Regression, KNN, Decision Tree)
  4. Model evaluation (Accuracy, Precision, Recall, F1 score, Confusion Matrix)
  5. Automatic best model selection (highest F1 score)
  6. Model persistence to saved_models/
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

# ── Feature & Label Constants ────────────────────────────────────────────────
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

RANDOM_SEED = 42
TOTAL_ROWS = 1200


# ── Step 1: Synthetic Data Generation ─────────────────────────────────────────
def generate_dataset(n=TOTAL_ROWS, seed=RANDOM_SEED):
    """Generates synthetic student performance data with realistic relationships."""
    np.random.seed(seed)
    
    attendance          = np.clip(np.random.normal(75, 15, n), 0, 100).round(1)
    internal_marks      = np.clip(np.random.normal(65, 18, n), 0, 100).round(1)
    assignment_score    = np.clip(np.random.normal(70, 15, n), 0, 100).round(1)
    previous_gpa        = np.clip(np.random.normal(7.0, 1.5, n), 0, 10).round(2)
    study_hours         = np.clip(np.random.normal(4, 2, n), 0, 14).round(1)
    backlogs            = np.random.choice([0, 0, 0, 1, 1, 2, 3, 4], size=n)
    class_participation = np.clip(np.random.normal(65, 20, n), 0, 100).round(1)

    # Weighted academic performance formula
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

    # Categorize into 3 risk levels
    risk_level = []
    for s in score:
        if s > 70:
            risk_level.append("LOW")
        elif s > 50:
            risk_level.append("MEDIUM")
        else:
            risk_level.append("HIGH")

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

    # Add realistic ~2% missing data
    rng = np.random.default_rng(seed)
    for col in ["attendance", "study_hours", "class_participation"]:
        missing_mask = rng.random(n) < 0.02
        df.loc[missing_mask, col] = np.nan

    return df


# ── Step 2: Preprocessing Pipelines ──────────────────────────────────────────
def build_scaled_pipeline():
    """Pipeline for distance and gradient-sensitive models (LR & KNN)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_unscaled_pipeline():
    """Pipeline for tree-based models (Decision Tree)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])


def load_and_prepare(csv_path):
    """Loads CSV dataset and prepares feature matrix X and target labels y."""
    df = pd.read_csv(csv_path).dropna(subset=["risk_level"])
    X = df[FEATURES]
    y = df["risk_level"].map(LABEL_MAP).values
    return X, y


# ── Step 3: Evaluation Metrics ────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """Computes standard classification evaluation metrics."""
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
    """Selects the best performing model based on F1 score."""
    return max(all_metrics, key=lambda m: (all_metrics[m]["f1_score"], all_metrics[m]["accuracy"]))


# ── Step 4: Full Training Runner ──────────────────────────────────────────────
def train_all_models(base_dir=None):
    """Trains all 3 models, saves serialized artifacts, and returns summary."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "saved_models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "student_data.csv")
    if not os.path.exists(csv_path):
        print(f"Generating dataset at {csv_path}...")
        df = generate_dataset()
        df.to_csv(csv_path, index=False)

    X, y = load_and_prepare(csv_path)
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
        joblib.dump(pipeline, os.path.join(models_dir, filename))

    best_model_name = select_best_model(all_metrics)

    summary = {
        "best_model": best_model_name,
        "models": all_metrics,
    }

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training completed successfully! Best model: {best_model_name}")
    return summary


if __name__ == "__main__":
    train_all_models()
