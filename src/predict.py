"""
predict.py
----------
Loads the trained pipeline and makes predictions on new customer data.

Run as a demo:
    python src/predict.py
"""

import os
import joblib
import pandas as pd
from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "..", "model", "churn_model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_churn(customer: dict) -> dict:
    """
    customer: dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES
    Returns: {"churn": "Yes"/"No", "churn_probability": float}
    """
    model = get_model()
    df = pd.DataFrame([customer])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = model.predict_proba(df)[0, 1]
    pred = "Yes" if proba >= 0.5 else "No"
    return {"churn": pred, "churn_probability": round(float(proba), 4)}


if __name__ == "__main__":
    sample_customer = {
        "tenure": 2,
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "SeniorCitizen": 0,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    result = predict_churn(sample_customer)
    print("Sample prediction (new, high-risk customer):")
    print(result)

    loyal_customer = {
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer",
    }
    result2 = predict_churn(loyal_customer)
    print("\nSample prediction (loyal, low-risk customer):")
    print(result2)
