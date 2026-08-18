# Customer Churn Prediction — End-to-End ML Project

A complete, runnable machine learning project: synthetic data generation,
preprocessing, model training with hyperparameter search, evaluation,
and a Flask web app + REST API for serving predictions.

## Project structure

```
churn_prediction_project/
├── README.md
├── requirements.txt
├── app.py                     # Flask web app + REST API
├── data/
│   ├── generate_data.py       # Creates the synthetic dataset
│   └── customer_churn.csv     # Generated dataset (5,000 customers)
├── src/
│   ├── preprocess.py          # Shared preprocessing pipeline (train + inference)
│   ├── train.py                # Trains model, evaluates, saves artifacts
│   └── predict.py              # Loads model, scores new customers
├── model/                     # Created after training
│   ├── churn_model.joblib     # Trained pipeline (preprocessing + classifier)
│   ├── metrics.json           # Evaluation metrics
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── templates/
    └── index.html             # Web UI form
```

## What it does

Predicts whether a telecom customer will **churn** (cancel their
subscription) based on their account profile — tenure, contract type,
monthly charges, services subscribed, etc. This mirrors the classic
"Telco Customer Churn" problem used widely in industry and interviews.

- **Model**: Random Forest classifier, tuned via `GridSearchCV`
  (n_estimators, max_depth, min_samples_leaf) with 4-fold CV on ROC-AUC.
- **Pipeline**: `ColumnTransformer` combining median-imputation +
  scaling for numeric features and most-frequent-imputation +
  one-hot encoding for categorical features, wrapped in a single
  sklearn `Pipeline` so training and inference always match.
- **Class imbalance**: handled with `class_weight="balanced"`.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset
```bash
python data/generate_data.py
```
Creates `data/customer_churn.csv` (5,000 synthetic customers, ~38% churn rate).
A CSV is already included, so you can skip this if you just want to run
the existing pipeline — but regenerate it if you want a fresh sample.

### 2. Train the model
```bash
python src/train.py
```
This will:
- Split data 80/20 (stratified)
- Run a grid search over Random Forest hyperparameters
- Print accuracy, precision, recall, F1, and ROC-AUC
- Save the trained pipeline to `model/churn_model.joblib`
- Save a confusion matrix and feature-importance chart as PNGs

Typical result on the synthetic data: **~75% ROC-AUC**, ~70% accuracy,
~70% recall on the churn class (tuned to catch at-risk customers,
which matters more than raw accuracy in a retention use case).

### 3. Run predictions from Python
```bash
python src/predict.py
```
Runs two example customers (a high-risk new customer and a low-risk
loyal one) through the trained model and prints churn probability.

You can also import it directly:
```python
from src.predict import predict_churn

result = predict_churn({
    "tenure": 3, "MonthlyCharges": 89.5, "TotalCharges": 268.5,
    "SeniorCitizen": 0, "gender": "Female", "Partner": "No",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "TechSupport": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
})
# -> {"churn": "Yes", "churn_probability": 0.87}
```

### 4. Run the web app + API
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in a browser for a form-based UI,
or call the REST API directly:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    "SeniorCitizen": 0, "gender": "Female", "Partner": "No",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "TechSupport": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
  }'
# -> {"churn": "Yes", "churn_probability": 0.8735}
```

`GET /api/health` returns `{"status": "ok"}` for uptime checks.

## Extending this project

- Swap `RandomForestClassifier` for `XGBoost`/`LightGBM` for a
  likely accuracy bump — the pipeline structure won't need to change.
- Replace `data/customer_churn.csv` with a real dataset (e.g. the
  IBM Telco Customer Churn dataset on Kaggle) — same column names
  work out of the box if you rename to match, or edit
  `src/preprocess.py`'s feature lists.
- Add SHAP values for per-prediction explainability.
- Containerize with Docker and deploy `app.py` behind `gunicorn` for
  production (the built-in Flask server is dev-only).
- Add MLflow or a simple `model/metrics_history.json` log to track
  performance across retraining runs.

## Notes

- The dataset here is **synthetically generated** (see
  `data/generate_data.py`) with realistic-but-artificial relationships
  between features and churn, purely for demonstration. Swap in your
  own data before drawing real business conclusions.
