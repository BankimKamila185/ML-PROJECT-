# Evaluation module
# This file calculates how well each ML model performs

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# The three classes our model predicts
CLASSES = ["LOW", "MEDIUM", "HIGH"]


def compute_metrics(y_true, y_pred):
    """
    Calculate 4 performance metrics for a model:

    - Accuracy  : Overall % of correct predictions
    - Precision : Of all predicted positives, how many are actually correct
    - Recall    : Of all actual positives, how many did the model find
    - F1 Score  : Balance between Precision and Recall (main metric)
    - Confusion Matrix : Table showing correct vs wrong predictions per class
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Confusion matrix: rows = actual class, columns = predicted class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    return {
        "accuracy":         round(float(accuracy),  4),
        "precision":        round(float(precision), 4),
        "recall":           round(float(recall),    4),
        "f1_score":         round(float(f1),        4),
        "confusion_matrix": cm.tolist(),
        "class_labels":     CLASSES,
    }


def select_best_model(all_metrics):
    """
    Look at all models and return the name of the best one.
    Best = highest F1 score.
    F1 score is chosen because it balances precision and recall.
    """
    best_name = None
    best_f1   = -1

    for model_name, metrics in all_metrics.items():
        if metrics["f1_score"] > best_f1:
            best_f1   = metrics["f1_score"]
            best_name = model_name

    return best_name
