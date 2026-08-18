"""
generate_data.py
-----------------
Generates a realistic synthetic "Telco Customer Churn" style dataset
and saves it to data/customer_churn.csv

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

def generate_dataset(n=N):
    gender = np.random.choice(["Male", "Female"], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    tenure = np.random.randint(0, 73, n)  # months with company

    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )

    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n).clip(min=0), 2)
    total_charges = total_charges.clip(min=0)

    tech_support = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])
    online_security = np.random.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22])

    # --- Build churn probability from a realistic underlying signal ---
    churn_prob = np.full(n, 0.10)

    churn_prob += (contract == "Month-to-month") * 0.30
    churn_prob += (contract == "One year") * 0.05
    churn_prob -= (contract == "Two year") * 0.08

    churn_prob += (internet_service == "Fiber optic") * 0.12
    churn_prob -= (internet_service == "No") * 0.05

    churn_prob += (tenure < 6) * 0.20
    churn_prob -= (tenure > 48) * 0.15

    churn_prob += (monthly_charges > 80) * 0.10
    churn_prob += (payment_method == "Electronic check") * 0.10

    churn_prob += (tech_support == "No") * 0.07
    churn_prob += (online_security == "No") * 0.07

    churn_prob += (senior_citizen == 1) * 0.05
    churn_prob -= (partner == "Yes") * 0.05
    churn_prob -= (dependents == "Yes") * 0.05

    churn_prob = churn_prob.clip(0.02, 0.95)
    churn = np.random.binomial(1, churn_prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["Churn"].value_counts(normalize=True))
