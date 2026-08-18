"""
train.py
--------
Trains a churn-prediction model, evaluates it, and saves the fitted
pipeline (preprocessing + model) to model/churn_model.joblib

Run:
    python src/train.py
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..")
DATA_PATH = os.path.join(_ROOT, "data", "customer_churn.csv")
MODEL_PATH = os.path.join(_ROOT, "model", "churn_model.joblib")
METRICS_PATH = os.path.join(_ROOT, "model", "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(_ROOT, "model", "confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = os.path.join(_ROOT, "model", "feature_importance.png")


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def main():
    print("Loading data...")
    df = load_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    preprocessor = build_preprocessor()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])

    # Small hyperparameter search — keep it fast but non-trivial
    param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, 12, None],
        "classifier__min_samples_leaf": [1, 2],
    }

    print("Running grid search (this may take a moment)...")
    grid = GridSearchCV(
        pipeline, param_grid, cv=4, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    # --- Evaluate ---
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_params": grid.best_params_,
    }

    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        if k != "best_params":
            print(f"{k:>10}: {v:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nSaved confusion matrix to {CONFUSION_MATRIX_PATH}")

    # --- Feature importance (top 15) ---
    ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_names
    importances = best_model.named_steps["classifier"].feature_importances_
    fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(7, 5))
    sns.barplot(x=fi.values, y=fi.index, color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    print(f"Saved feature importance plot to {FEATURE_IMPORTANCE_PATH}")

    # --- Save artifacts ---
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved trained pipeline to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
