# Dataset Generator
# Creates a realistic student performance dataset with 1200 rows
# Uses numpy to generate random numbers with realistic distributions

import numpy as np
import pandas as pd
import os

# Fix random seed so results are the same every time we run this
RANDOM_SEED  = 42
TOTAL_ROWS   = 1200

np.random.seed(RANDOM_SEED)


def generate_dataset(n=TOTAL_ROWS):
    # ── Step 1: Generate random values for each feature ──────────────────
    # np.random.normal(mean, std_deviation, count) gives realistic spread
    attendance          = np.clip(np.random.normal(75, 15, n), 0, 100).round(1)
    internal_marks      = np.clip(np.random.normal(65, 18, n), 0, 100).round(1)
    assignment_score    = np.clip(np.random.normal(70, 15, n), 0, 100).round(1)
    previous_gpa        = np.clip(np.random.normal(7.0, 1.5, n), 0, 10).round(2)
    study_hours         = np.clip(np.random.normal(4, 2, n), 0, 14).round(1)
    backlogs            = np.random.choice([0, 0, 0, 1, 1, 2, 3, 4], size=n)
    class_participation = np.clip(np.random.normal(65, 20, n), 0, 100).round(1)

    # ── Step 2: Calculate a risk score (0 to 100) ────────────────────────
    # Higher score = better performance = lower risk
    score = (
        0.25 * attendance
        + 0.20 * internal_marks
        + 0.15 * assignment_score
        + 0.20 * (previous_gpa / 10 * 100)   # convert GPA to 0-100 scale
        + 0.10 * (study_hours / 14 * 100)     # convert hours to 0-100 scale
        + 0.05 * class_participation
        - 5   * backlogs                       # each backlog reduces score
    )
    score = np.clip(score, 0, 100)

    # Add a little random noise to make the data realistic
    score = score + np.random.normal(0, 3, n)
    score = np.clip(score, 0, 100)

    # ── Step 3: Assign risk labels based on score ─────────────────────────
    # score > 70  → LOW risk
    # 50 to 70    → MEDIUM risk
    # below 50    → HIGH risk
    risk_level = []
    for s in score:
        if s > 70:
            risk_level.append("LOW")
        elif s > 50:
            risk_level.append("MEDIUM")
        else:
            risk_level.append("HIGH")

    # ── Step 4: Build the DataFrame ───────────────────────────────────────
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

    # Add ~2% missing values to make data realistic (like real datasets)
    rng = np.random.default_rng(RANDOM_SEED)
    for col in ["attendance", "study_hours", "class_participation"]:
        missing_mask = rng.random(n) < 0.02
        df.loc[missing_mask, col] = np.nan

    return df


if __name__ == "__main__":
    # Save the dataset to the data/ folder
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "student_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = generate_dataset()
    df.to_csv(out_path, index=False)

    print(f"Dataset saved to: {out_path}")
    print(f"Total rows: {len(df)}")
    print("\nRisk level distribution:")
    print(df["risk_level"].value_counts())
