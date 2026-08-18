"""
preprocess.py
-------------
Defines the preprocessing pipeline: which columns are numeric vs
categorical, and how to transform them. Used by both train.py and
predict.py so training and inference always match.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "TechSupport", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

TARGET = "Churn"

DROP_COLUMNS = ["customerID"]


def build_preprocessor() -> ColumnTransformer:
    """Returns a ColumnTransformer that scales numeric features and
    one-hot encodes categorical features. Missing values are imputed."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor
