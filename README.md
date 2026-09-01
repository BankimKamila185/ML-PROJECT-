# Student Performance Risk Prediction Using Classification Algorithms

> **Module V – Supervised Learning: Classification** | CO3 Machine Learning Fundamentals

---

## Problem Statement

Academic institutions often fail to identify at-risk students early enough to intervene effectively. This project builds a Machine Learning system that predicts a student's academic risk level (**LOW**, **MEDIUM**, or **HIGH**) based on easily measurable academic features, enabling timely intervention.

This is a **classification problem** because the output is a discrete category, not a continuous value.

---

## Objectives

1. Implement three classification algorithms: Logistic Regression, KNN, Decision Tree
2. Train and evaluate all models on a student performance dataset
3. Automatically select the best model by F1 score
4. Expose predictions via a REST API (FastAPI)
5. Present results through a professional web dashboard (React + TypeScript + Tailwind)
6. Store prediction history in a SQLite database

---

## Technologies

| Layer | Technology |
|---|---|
| **Frontend** | React + TypeScript + Tailwind CSS + Recharts |
| **Backend** | Python + FastAPI + Pydantic |
| **Machine Learning** | Scikit-learn + NumPy + Pandas + Joblib |
| **Database** | SQLite + SQLAlchemy |

---

## ML Algorithms

### 1. Logistic Regression
Despite its name, Logistic Regression is a **classification** algorithm. It models the probability of each class using a logistic (sigmoid) function, extended to multiple classes via the softmax function (multinomial logistic regression). It works well when the decision boundary is approximately linear.

**Hyperparameters used:**
- `max_iter=1000` (enough iterations for convergence)
- `solver='lbfgs'` (suitable for multinomial)
- `C=1.0` (inverse regularization strength)

### 2. K-Nearest Neighbors (KNN)
KNN is a **non-parametric** algorithm. To classify a new sample, it finds the `k` most similar training samples (by Euclidean distance) and assigns the majority class. It requires feature scaling because it is distance-based.

**Hyperparameters used:**
- `n_neighbors=7`
- `metric='minkowski'`, `p=2` (Euclidean distance)

### 3. Decision Tree
A Decision Tree recursively partitions the feature space by asking binary questions (e.g., "Is attendance < 60?"). It learns a tree of if-else rules. It does NOT require feature scaling and is highly interpretable.

**Hyperparameters used:**
- `max_depth=10` (prevents overfitting)
- `min_samples_split=5`
- `criterion='gini'` (Gini impurity for splits)

---

## Dataset

### Features

| Feature | Range | Description |
|---|---|---|
| `attendance` | 0–100 | Lecture attendance percentage |
| `internal_marks` | 0–100 | Internal examination score |
| `assignment_score` | 0–100 | Assignment submission score |
| `previous_gpa` | 0–10 | Previous semester GPA |
| `study_hours` | 0–24 | Daily study hours |
| `backlogs` | 0+ | Number of failed/pending subjects |
| `class_participation` | 0–100 | Classroom participation percentage |

### Target

`risk_level` ∈ {**LOW**, **MEDIUM**, **HIGH**}

### Data Generation

A synthetic dataset of 1,200 rows is generated using `numpy.random.seed(42)` for reproducibility. Risk labels are assigned via a weighted scoring function (not randomly), so the models learn real patterns:

```
score = 0.25 × attendance + 0.20 × internal_marks + 0.15 × assignment_score
      + 0.20 × (GPA/10×100) + 0.10 × (study_hours/14×100)
      + 0.05 × class_participation − 5 × backlogs
```

- score > 70 → **LOW**
- 50 < score ≤ 70 → **MEDIUM**
- score ≤ 50 → **HIGH**

---

## Methodology

```
Synthetic Dataset (1200 rows)
        ↓
Data Preprocessing
  • Missing value imputation (median)
  • Feature scaling (StandardScaler for LR & KNN)
  • Label encoding: LOW=0, MEDIUM=1, HIGH=2
        ↓
Train/Test Split (80% / 20%, stratified)
        ↓
Model Training
  • Logistic Regression
  • K-Nearest Neighbors
  • Decision Tree
  (each wrapped in sklearn Pipeline)
        ↓
Model Evaluation
  • Accuracy, Precision, Recall, F1 Score
  • Confusion Matrix
        ↓
Best Model Selection (highest F1 Score)
        ↓
Save models with joblib
        ↓
FastAPI Prediction API
        ↓
React Web Dashboard
```

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | TP + TN / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many are correct |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many were found |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |

All metrics are computed with `average='weighted'` to account for class imbalance.

### Confusion Matrix

A confusion matrix is an N×N table where rows = actual classes and columns = predicted classes. Diagonal values are correct predictions. Off-diagonal values are misclassifications.

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm 9+

### 1. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train ML Models

```bash
# From backend directory, with venv activated
python app/ml/train.py
```

This will:
1. Generate the synthetic dataset
2. Train all 3 classification models
3. Evaluate and compare them
4. Select the best model by F1 score
5. Save all models and metrics to `saved_models/`

### 3. Start Backend

```bash
# From backend directory, with venv activated
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard: http://localhost:5173

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Predict risk level |
| GET | `/api/models` | Model metrics & best model |
| GET | `/api/metrics` | Alias for /models |
| GET | `/api/history` | Prediction history |
| GET | `/api/dashboard` | Dashboard stats |

---

## Project Structure

```
ML PROJECT/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── schemas.py           # Pydantic models
│   │   ├── routes/              # API route handlers
│   │   ├── services/            # Business logic
│   │   ├── ml/                  # ML training & inference
│   │   └── database/            # SQLAlchemy models
│   ├── data/                    # student_data.csv
│   ├── saved_models/            # Trained joblib pipelines + metrics.json
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/               # Dashboard, Predict, Comparison, Matrix, History
│   │   ├── components/          # Sidebar, RiskBadge
│   │   ├── api/                 # Axios client
│   │   └── types/               # TypeScript interfaces
│   ├── tailwind.config.js
│   └── package.json
└── README.md
```

---

## Why Best Model is Selected Automatically

After training, all three models are evaluated on the held-out test set. The model with the highest **F1 score** is automatically selected:

```python
best_model = max(metrics, key=lambda m: (metrics[m]['f1_score'], metrics[m]['accuracy']))
```

F1 score is preferred over accuracy because it balances precision and recall, which is important for imbalanced class distributions (e.g., fewer HIGH-risk students).

---

## Future Improvements

- Add cross-validation (k-fold) for more robust model evaluation
- Implement hyperparameter tuning with GridSearchCV
- Add email alerts for HIGH RISK predictions
- Support upload of real institutional datasets via CSV
- Integrate explainability (SHAP values for feature importance)
