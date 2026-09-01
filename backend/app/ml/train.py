"""
Training Script - Student Performance Risk Prediction
Module V: Supervised Learning - Classification

This script:
  1. Generates the dataset (if not already present)
  2. Splits data into training and testing sets
  3. Trains 3 classification models
  4. Evaluates each model
  5. Picks the best model automatically
  6. Saves all models to disk

Run this from the backend folder:
    python app/ml/train.py
"""

import os
import sys
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Add the app folder to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.ml.generate_data  import generate_dataset
from app.ml.preprocessing  import load_and_prepare, build_scaled_pipeline, build_unscaled_pipeline
from app.ml.evaluate       import compute_metrics, select_best_model

# ─── File paths ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "..", "data", "student_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "saved_models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)


# ─── Step 1: Create dataset if it does not exist ─────────────────────────────
print("=" * 55)
print("Student Performance Risk Prediction - Training")
print("Module V: Supervised Learning - Classification")
print("=" * 55)

if not os.path.exists(DATA_PATH):
    print("\nDataset not found. Generating 1200 rows...")
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved: {DATA_PATH}")
else:
    print(f"\nDataset found: {DATA_PATH}")


# ─── Step 2: Load data and split into train / test ───────────────────────────
X, y = load_and_prepare(DATA_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 80% train, 20% test
    random_state=42,      # fixed seed for reproducibility
    stratify=y            # keep class balance same in both splits
)

print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")


# ─── Step 3: Define the three classification models ───────────────────────────
# Each model is wrapped in a Pipeline so preprocessing is included automatically.
# This prevents data leakage (test data is never used to fit the scaler).

models = {

    # Model 1: Logistic Regression
    # - Works by finding the best boundary between classes
    # - Needs scaled features (StandardScaler)
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", build_scaled_pipeline()),
        ("classifier",   LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42
        )),
    ]),

    # Model 2: K-Nearest Neighbors (KNN)
    # - Finds the 7 most similar training samples and votes
    # - Needs scaled features because it uses distance
    "KNN": Pipeline(steps=[
        ("preprocessor", build_scaled_pipeline()),
        ("classifier",   KNeighborsClassifier(
            n_neighbors=7,
            metric="minkowski",
            p=2
        )),
    ]),

    # Model 3: Decision Tree
    # - Learns a tree of if-else rules from data
    # - Does NOT need scaling
    "Decision Tree": Pipeline(steps=[
        ("preprocessor", build_unscaled_pipeline()),
        ("classifier",   DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            criterion="gini",
            random_state=42
        )),
    ]),
}


# ─── Step 4: Train and evaluate each model ───────────────────────────────────
all_metrics = {}

for model_name, pipeline in models.items():
    print(f"\nTraining {model_name}...")

    # Train the model on training data
    pipeline.fit(X_train, y_train)

    # Make predictions on test data (model has NOT seen this before)
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    metrics = compute_metrics(y_test, y_pred)
    all_metrics[model_name] = metrics

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")


# ─── Step 5: Pick the best model ─────────────────────────────────────────────
best_model_name = select_best_model(all_metrics)

print("\n" + "=" * 55)
print(f"Best Model (highest F1 score): {best_model_name}")
print("=" * 55)


# ─── Step 6: Save all models and metrics to disk ─────────────────────────────
for model_name, pipeline in models.items():
    # Make a safe filename: "Logistic Regression" → "logistic_regression.joblib"
    filename  = model_name.lower().replace(" ", "_") + ".joblib"
    save_path = os.path.join(MODELS_DIR, filename)
    joblib.dump(pipeline, save_path)
    print(f"Saved: {save_path}")

# Save metrics and best model name to a JSON file
summary = {
    "best_model": best_model_name,
    "models":     all_metrics,
}

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved: {metrics_path}")
print("\nTraining complete!")
