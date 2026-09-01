# Preprocessing module
# This file handles cleaning and preparing data before training the ML model

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# These are the 7 input features used for prediction
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_score",
    "previous_gpa",
    "study_hours",
    "backlogs",
    "class_participation",
]

# Convert text labels to numbers that the ML model can understand
# LOW = 0, MEDIUM = 1, HIGH = 2
LABEL_MAP = {
    "LOW":    0,
    "MEDIUM": 1,
    "HIGH":   2
}

# Reverse: convert numbers back to text labels
REVERSE_LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def encode_labels(series):
    """Convert LOW/MEDIUM/HIGH text to 0/1/2 numbers"""
    return series.map(LABEL_MAP).values


def decode_labels(arr):
    """Convert 0/1/2 numbers back to LOW/MEDIUM/HIGH text"""
    return [REVERSE_LABEL_MAP[int(x)] for x in arr]


def build_scaled_pipeline():
    """
    Pipeline for Logistic Regression and KNN.
    Step 1: Fill in any missing values with the median
    Step 2: Scale all features to same range (StandardScaler)
    KNN and LR are sensitive to feature scale, so this is important.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def build_unscaled_pipeline():
    """
    Pipeline for Decision Tree.
    Step 1: Fill in any missing values with the median
    Decision Tree does NOT need scaling, so we skip that step.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])


def load_and_prepare(csv_path):
    """
    Load the CSV file and return X (features) and y (labels)
    X = the input columns
    y = the risk_level column converted to numbers
    """
    df = pd.read_csv(csv_path)

    # Remove rows where the target (risk_level) is missing
    df = df.dropna(subset=["risk_level"])

    X = df[FEATURES]
    y = encode_labels(df["risk_level"])

    return X, y
