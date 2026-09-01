# 📚 Complete Project Architecture & Code Explanation Guide

**Project Title:** Student Performance Risk Prediction Using Classification Algorithms  
**Academic Module:** Module V – Supervised Learning: Classification (CO3)  

---

## 📑 Table of Contents
1. [System Architecture & Workflow](#1-system-architecture--workflow)
2. [Project Folder Structure Overview](#2-project-folder-structure-overview)
3. [Machine Learning (ML) Engine Files](#3-machine-learning-ml-engine-files)
   - [`generate_data.py`](#31-generate_datapy)
   - [`preprocessing.py`](#32-preprocessingpy)
   - [`evaluate.py`](#33-evaluatepy)
   - [`train.py`](#34-trainpy)
   - [`predict.py`](#35-predictpy)
4. [Backend (FastAPI) Files](#4-backend-fastapi-files)
   - [`main.py`](#41-mainpy)
   - [`schemas.py`](#42-schemaspy)
   - [`database/models.py`](#43-databasemodelspy)
   - [`database/session.py`](#44-databasesessionpy)
   - [`services/ml_service.py`](#45-servicesml_servicepy)
   - [`services/db_service.py`](#46-servicesdb_servicepy)
   - [`routes/predict.py`](#47-routespredictpy)
   - [`routes/models.py`](#48-routesmodelspy)
   - [`routes/history.py`](#49-routeshistorypy)
5. [Frontend (React + TypeScript) Files](#5-frontend-react--typescript-files)
   - [`api/client.ts`](#51-apiclientts)
   - [`pages/Dashboard.tsx`](#52-pagesdashboardtsx)
   - [`pages/Prediction.tsx`](#53-pagespredictiontsx)
   - [`pages/ModelComparison.tsx`](#54-pagesmodelcomparisontsx)
   - [`pages/ConfusionMatrix.tsx`](#55-pagesconfusionmatrixtsx)
   - [`pages/History.tsx`](#56-pageshistorytsx)
6. [Step-by-Step Data Flow (What Happens When You Click "Predict")](#6-step-by-step-data-flow)
7. [Viva & Examiner Q&A Cheat Sheet](#7-viva--examiner-qa-cheat-sheet)

---

## 1. System Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│               1. React Frontend (Vite)                 │
│  User inputs student scores & attendance on Dashboard  │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST /api/predict (JSON)
                           ▼
┌────────────────────────────────────────────────────────┐
│             2. FastAPI Backend Application             │
│  • Pydantic validates input ranges (0-100, 0-10, etc.) │
│  • Calls ML Service                                    │
└─────────────┬────────────────────────────┬─────────────┘
              │ Passes data                │ Saves history
              ▼                            ▼
┌───────────────────────────────┐ ┌──────────────────────┐
│  3. Trained ML Pipeline       │ │ 4. SQLite Database   │
│  • SimpleImputer (median)     │ │ Stores prediction    │
│  • StandardScaler (if needed) │ │ timestamp, risk      │
│  • Selected Model Predicts:   │ │ score, confidence    │
│    (LOW / MEDIUM / HIGH)      │ └──────────────────────┘
└─────────────┬─────────────────┘
              │ Returns Risk Level + Confidence
              ▼
┌────────────────────────────────────────────────────────┐
│               5. Response to Frontend                  │
│  Displays: Risk Badge, Confidence %, Rule Advice       │
└────────────────────────────────────────────────────────┘
```

---

## 2. Project Folder Structure Overview

```
ML PROJECT/
├── backend/
│   ├── app/
│   │   ├── main.py              # Entry point for FastAPI backend
│   │   ├── schemas.py           # Pydantic data validation classes
│   │   ├── database/            # SQLite database models and connection
│   │   │   ├── models.py        # Database table schema (columns)
│   │   │   └── session.py       # DB engine & connection session
│   │   ├── ml/                  # Core Machine Learning scripts
│   │   │   ├── generate_data.py # Creates the 1,200 student dataset
│   │   │   ├── preprocessing.py # Scaling, imputation, label encoders
│   │   │   ├── evaluate.py      # Accuracy, Precision, Recall, F1, Confusion Matrix
│   │   │   ├── train.py         # Trains LR, KNN, Decision Tree & picks the best
│   │   │   └── predict.py       # Single-input inference helper
│   │   ├── routes/              # API Endpoints
│   │   │   ├── predict.py       # POST /api/predict
│   │   │   ├── models.py        # GET /api/models & /api/metrics
│   │   │   └── history.py       # GET /api/history & /api/dashboard
│   │   └── services/            # Business Logic & Database queries
│   │       ├── ml_service.py    # Loads saved model into RAM and predicts
│   │       └── db_service.py    # Database CRUD (Create, Read) functions
│   ├── data/
│   │   └── student_data.csv     # Generated training dataset
│   ├── saved_models/            # Joblib files & model evaluation metrics
│   │   ├── logistic_regression.joblib
│   │   ├── knn.joblib
│   │   ├── decision_tree.joblib
│   │   └── metrics.json
│   ├── requirements.txt         # Python libraries needed
│   └── student_predictions.db   # SQLite database file
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx             # React DOM root render
│   │   ├── App.tsx              # Router navigation for all pages
│   │   ├── api/client.ts        # Axios API client functions
│   │   ├── types/index.ts       # TypeScript interfaces for API data
│   │   ├── components/
│   │   │   ├── Sidebar.tsx      # Left navigation bar
│   │   │   └── RiskBadge.tsx    # Colorful badge component (LOW/MEDIUM/HIGH)
│   │   └── pages/
│   │       ├── Dashboard.tsx        # Overview KPI statistics
│   │       ├── Prediction.tsx       # Student risk prediction form
│   │       ├── ModelComparison.tsx  # Charts & tables comparing all 3 models
│   │       ├── ConfusionMatrix.tsx  # Matrix grid visualizer
│   │       └── History.tsx          # Previous prediction table
│   ├── package.json             # Frontend dependencies
│   └── tailwind.config.js       # Tailwind CSS styling configuration
│
└── README.md                    # Academic project documentation
```

---

## 3. Machine Learning (ML) Engine Files

### 3.1 `generate_data.py`
* **File Location:** `backend/app/ml/generate_data.py`
* **What is it for?**  
  Generates a synthetic but statistically realistic dataset of 1,200 students with realistic academic parameters and assigned risk categories.
* **Core Code Explanation:**
  - `np.random.normal(mean, std_dev, n)`: Creates normal bell-curve distributions for marks, attendance, and study hours.
  - `score formula`: Computes a weighted academic score:
    $$\text{Score} = 0.25 \times \text{attendance} + 0.20 \times \text{internals} + 0.15 \times \text{assignments} + 0.20 \times (\text{GPA} \times 10) + 0.10 \times \text{study\_hours} + 0.05 \times \text{participation} - 5 \times \text{backlogs}$$
  - **Risk assignment loop:**
    - `Score > 70` $\rightarrow$ **LOW RISK**
    - `50 < Score <= 70` $\rightarrow$ **MEDIUM RISK**
    - `Score <= 50` $\rightarrow$ **HIGH RISK**
  - **Missing Values:** Injects 2% missing values (`NaN`) to test data-cleaning/imputation in real-world scenarios.

---

### 3.2 `preprocessing.py`
* **File Location:** `backend/app/ml/preprocessing.py`
* **What is it for?**  
  Cleans missing values, scales features, and encodes target classes.
* **Core Code Explanation:**
  - `LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}`: Converts category strings to integer numbers for Scikit-learn.
  - `build_scaled_pipeline()`:
    1. `SimpleImputer(strategy="median")`: Fills any missing `NaN` values with the column median.
    2. `StandardScaler()`: Standardizes features ($z = \frac{x - \mu}{\sigma}$). Required for **Logistic Regression** and **KNN** because distance/gradient algorithms are sensitive to differing scales.
  - `build_unscaled_pipeline()`:
    - Uses only `SimpleImputer`. **Decision Trees** do not require scaling because rule splits ($x > c$) are invariant to monotonic transformations.

---

### 3.3 `evaluate.py`
* **File Location:** `backend/app/ml/evaluate.py`
* **What is it for?**  
  Calculates all test evaluation metrics and automatically selects the best performing model.
* **Core Code Explanation:**
  - `accuracy_score(y_true, y_pred)`: Total correct / Total predictions.
  - `precision_score(..., average="weighted")`: $\frac{TP}{TP + FP}$ weighted across all 3 classes.
  - `recall_score(..., average="weighted")`: $\frac{TP}{TP + FN}$ weighted across all 3 classes.
  - `f1_score(..., average="weighted")`: Harmonic mean of Precision and Recall:
    $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - `confusion_matrix(y_true, y_pred)`: Creates the $3 \times 3$ matrix of actual vs predicted counts.
  - `select_best_model(all_metrics)`: Iterates through all models and picks the one with the **highest $F_1$ score**.

---

### 3.4 `train.py`
* **File Location:** `backend/app/ml/train.py`
* **What is it for?**  
  The main training orchestrator script. Runs the entire pipeline: loading data, splitting, model training, evaluation, best model selection, and model persistence (`joblib`).
* **Core Code Explanation:**
  1. `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`: Splits 80% (960 rows) for training and 20% (240 rows) for testing. `stratify=y` ensures identical class proportions in both sets.
  2. **Model Definitions:**
     - **Logistic Regression:** Linear classifier using Softmax multi-class loss with `solver='lbfgs'`.
     - **K-Nearest Neighbors (KNN):** Distance-based voting classifier with `n_neighbors=7`.
     - **Decision Tree:** Rule-based tree with `max_depth=10` and `min_samples_split=5`.
  3. `pipeline.fit(X_train, y_train)`: Fits both the preprocessor and the classifier together to avoid data leakage.
  4. `joblib.dump(pipeline, save_path)`: Saves trained models into binary `.joblib` files inside `backend/saved_models/`.
  5. Saves `metrics.json` containing accuracy, precision, recall, F1, and confusion matrix data for the web UI.

---

### 3.5 `predict.py`
* **File Location:** `backend/app/ml/predict.py`
* **What is it for?**  
  Provides a standalone function `predict(input_data)` to test predictions from Python.
* **Core Code Explanation:**
  - Reads `metrics.json` to find the winning `best_model`.
  - Loads that model's `.joblib` pipeline.
  - Wraps input values in a Pandas DataFrame to preserve feature column names.
  - Calls `pipeline.predict(X)` and `pipeline.predict_proba(X)` to get the predicted risk level and confidence percentage.

---

## 4. Backend (FastAPI) Files

### 4.1 `main.py`
* **File Location:** `backend/app/main.py`
* **What is it for?**  
  The entry point of the FastAPI application. Sets up CORS, database startup, and routes.
* **Core Code Explanation:**
  - `@asynccontextmanager async def lifespan(app)`: Runs `init_db()` when the server boots up to ensure SQLite tables exist.
  - `CORSMiddleware`: Permits requests from the Vite React frontend (`http://localhost:5173`).
  - `app.include_router(...)`: Registers the prediction, models, and history routes under `/api`.
  - `/api/health`: Health check endpoint returning `{"status": "ok", "trained": true}`.

---

### 4.2 `schemas.py`
* **File Location:** `backend/app/schemas.py`
* **What is it for?**  
  Pydantic data schemas that validate incoming HTTP requests and format outgoing JSON responses.
* **Core Code Explanation:**
  - `PredictionRequest`: Enforces validation rules on every input field:
    - `attendance`: Float between `0.0` and `100.0`
    - `internal_marks`: Float between `0.0` and `100.0`
    - `assignment_score`: Float between `0.0` and `100.0`
    - `previous_gpa`: Float between `0.0` and `10.0`
    - `study_hours`: Float between `0.0` and `24.0`
    - `backlogs`: Integer $\ge 0$
    - `class_participation`: Float between `0.0` and `100.0`
  - Returns clear validation errors (HTTP 422) if invalid data is sent.

---

### 4.3 `database/models.py`
* **File Location:** `backend/app/database/models.py`
* **What is it for?**  
  Defines the SQLite database table structure using SQLAlchemy ORM.
* **Core Code Explanation:**
  - `class PredictionRecord(Base)`: Maps to table `prediction_history`.
  - Columns include: `id` (Primary Key), `student_id`, input values (`attendance`, `internal_marks`, etc.), output `risk_level`, `confidence`, `model_used`, and `created_at` (timestamp).

---

### 4.4 `database/session.py`
* **File Location:** `backend/app/database/session.py`
* **What is it for?**  
  Establishes the SQLite connection (`student_predictions.db`) and provides a database session dependency for FastAPI.
* **Core Code Explanation:**
  - `create_engine(..., connect_args={"check_same_thread": False})`: Creates the SQLite database connection safely for multithreaded web servers.
  - `get_db()`: Generator function that yields a database session and closes it after the request finishes.

---

### 4.5 `services/ml_service.py`
* **File Location:** `backend/app/services/ml_service.py`
* **What is it for?**  
  The bridge between FastAPI and the Machine Learning models. Loads model pipelines into memory once (singleton cache) to ensure low latency.
* **Core Code Explanation:**
  - `get_pipeline(model_name)`: Loads the `.joblib` file once and stores it in `_loaded_pipelines` dictionary.
  - `run_prediction(input_data)`: Runs model inference, extracts class probabilities, and attaches academic advice from the rule-based `RECOMMENDATIONS` map.

---

### 4.6 `services/db_service.py`
* **File Location:** `backend/app/services/db_service.py`
* **What is it for?**  
  Handles database CRUD (Create, Read) queries.
* **Core Code Explanation:**
  - `create_record(...)`: Creates a new row with a random student ID (e.g., `STU-A1B2C3D4`) and commits it to SQLite.
  - `get_history(...)`: Performs filtered queries with search keyword matching, risk level filter, pagination (`skip`, `limit`), and sorting (`asc`, `desc`).
  - `get_summary_stats(...)`: Counts total, low, medium, and high risk records for the Dashboard.

---

### 4.7 `routes/predict.py`
* **File Location:** `backend/app/routes/predict.py`
* **What is it for?**  
  Handles `POST /api/predict`. Receives student form data, calls `ml_service`, saves the result in SQLite, and returns the risk prediction.

---

### 4.8 `routes/models.py`
* **File Location:** `backend/app/routes/models.py`
* **What is it for?**  
  Handles `GET /api/models` and `GET /api/metrics`. Returns all trained model metrics, F1 scores, and confusion matrix coordinates.

---

### 4.9 `routes/history.py`
* **File Location:** `backend/app/routes/history.py`
* **What is it for?**  
  - `GET /api/history`: Returns paginated and filtered historical student predictions.
  - `GET /api/dashboard`: Returns aggregate count stats and best model metrics for dashboard cards.

---

## 5. Frontend (React + TypeScript) Files

### 5.1 `api/client.ts`
* **File Location:** `frontend/src/api/client.ts`
* **What is it for?**  
  Centralized Axios HTTP client communicating with backend endpoints (`http://localhost:8000`).
* **Functions:**
  - `checkHealth()`: Checks backend status.
  - `predict(data)`: Sends form data to `POST /api/predict`.
  - `getModels()`: Fetches comparison metrics from `GET /api/models`.
  - `getHistory(params)`: Queries `GET /api/history` with search and filters.
  - `getDashboard()`: Queries `GET /api/dashboard`.

---

### 5.2 `pages/Dashboard.tsx`
* **File Location:** `frontend/src/pages/Dashboard.tsx`
* **What is it for?**  
  Main executive overview showing KPI cards:
  - Total students analyzed
  - Count and % of Low, Medium, and High risk students
  - Best ML Model selected badge
  - Model Accuracy & F1-score badges

---

### 5.3 `pages/Prediction.tsx`
* **File Location:** `frontend/src/pages/Prediction.tsx`
* **What is it for?**  
  Interactive input form for student academic parameters:
  - **Academic Info:** Attendance, Internal Marks, Assignment Score, GPA, Backlogs.
  - **Study Info:** Study Hours, Class Participation.
  - **Validation:** Real-time client-side range checking.
  - **Result Card:** Displays predicted risk with colorful `RiskBadge`, confidence percentage, algorithm used, and recommendations.

---

### 5.4 `pages/ModelComparison.tsx`
* **File Location:** `frontend/src/pages/ModelComparison.tsx`
* **What is it for?**  
  Visual evaluation comparing Logistic Regression, KNN, and Decision Tree:
  - 4 interactive Recharts Bar Charts (Accuracy, Precision, Recall, F1 Score).
  - Summary comparison table highlighting the best model with a ⭐ Best badge.

---

### 5.5 `pages/ConfusionMatrix.tsx`
* **File Location:** `frontend/src/pages/ConfusionMatrix.tsx`
* **What is it for?**  
  Interactive $3 \times 3$ Confusion Matrix heatmaps:
  - Allows switching between Logistic Regression, KNN, and Decision Tree.
  - Color-codes correct predictions (blue diagonal) vs misclassifications (red off-diagonal).
  - Includes clear academic explanations of what rows and columns represent.

---

### 5.6 `pages/History.tsx`
* **File Location:** `frontend/src/pages/History.tsx`
* **What is it for?**  
  Historical audit log of all previous predictions:
  - Instant search by Student ID or Risk Level.
  - Filter dropdown for LOW, MEDIUM, or HIGH risk.
  - Sortable columns (Date, Attendance, Marks, etc.).
  - Clean pagination.

---

## 6. Step-by-Step Data Flow

Here is what happens behind the scenes during a single prediction:

1. **User enters data** on the React form (`Prediction.tsx`) and clicks **"PREDICT RISK"**.
2. **Client Validation:** React verifies that all numbers fall within allowable bounds (e.g., Attendance between 0–100).
3. **HTTP Request:** Axios sends a JSON payload to `http://localhost:8000/api/predict`.
4. **Server Validation:** FastAPI Pydantic schema (`schemas.py`) validates the payload structure.
5. **Model Inference:** `ml_service.py` feeds the 7 features into the trained Scikit-learn pipeline (`logistic_regression.joblib`).
6. **Feature Processing & Prediction:**
   - `SimpleImputer` replaces any missing inputs with training medians.
   - `StandardScaler` standardizes the input features.
   - Classifier predicts class `0` (LOW), `1` (MEDIUM), or `2` (HIGH) and calculates class probability.
7. **Database Persistence:** `db_service.py` generates a student ID (`STU-XXXX`) and inserts the record into SQLite (`student_predictions.db`).
8. **HTTP Response:** The backend returns JSON containing the risk level, confidence score, model name, and rule recommendation.
9. **UI Update:** React renders the result card with animations, and the Dashboard & History tables update automatically.

---

## 7. Viva & Examiner Q&A Cheat Sheet

### Q1: Why is this a Classification problem rather than Regression?
> **Answer:** In regression, the output is a continuous numerical value (e.g., predicting the exact percentage 76.4%). In our system, the output is a **discrete category** (LOW, MEDIUM, or HIGH risk). Therefore, it belongs to **Module V: Supervised Learning – Classification**.

### Q2: Why did you compare Logistic Regression, KNN, and Decision Tree?
> **Answer:** To evaluate linear (Logistic Regression), instance/distance-based (KNN), and non-linear rule-based (Decision Tree) classification paradigms on academic data, and to demonstrate key concepts from Module V.

### Q3: Why does Logistic Regression or KNN need Feature Scaling, but Decision Trees do not?
> **Answer:**  
> - **KNN** calculates Euclidean distances ($\sqrt{\sum (x_i - y_i)^2}$). Features with larger numerical ranges (e.g., Attendance 0–100 vs GPA 0–10) would dominate the distance calculation without scaling.  
> - **Logistic Regression** uses gradient descent for coefficient optimization, which converges faster and more reliably when features are on the same scale.  
> - **Decision Trees** split features independently using threshold inequalities ($x_j \le \theta$), so the scale of other features does not affect tree splits.

### Q4: Why select the best model based on F1 Score rather than Accuracy?
> **Answer:** Academic risk datasets often have class imbalance (e.g., fewer HIGH-risk students than MEDIUM/LOW). Accuracy can be misleading if a model predicts the majority class well but misses at-risk students. The **F1 Score** balances Precision and Recall, making it the more reliable metric.

### Q5: How do you prevent Data Leakage during preprocessing?
> **Answer:** We use Scikit-learn **Pipelines**. Preprocessors (such as `StandardScaler` and `SimpleImputer`) are fitted **only on the training split** (`X_train`) and then applied to transform test and live inference data without ever seeing test labels.

### Q6: Are the academic recommendations generated by the ML model?
> **Answer:** **No.** The ML model is responsible strictly for classification (predicting LOW, MEDIUM, or HIGH risk and confidence). The recommendations are rule-based advice mapped to each risk tier, keeping the ML prediction distinct and transparent.
