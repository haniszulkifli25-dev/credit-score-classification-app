"""Shared, reproducible feature processing for the portfolio prototype."""
import re
import numpy as np
import pandas as pd

IDENTIFIERS = ["ID", "Customer_ID", "Month", "Name", "SSN"]
NUMERIC = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]
INTERFACE = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_of_Loan",
    "Credit_History_Age", "Outstanding_Debt", "Credit_Utilization_Ratio"
]
LABELS = {0: "Poor", 1: "Standard", 2: "Good"}

def credit_months(value):
    if pd.isna(value):
        return np.nan
    s = str(value)
    years = re.search(r"(\d+)\s*Year", s)
    months = re.search(r"(\d+)\s*Month", s)
    if not years and not months:
        return np.nan
    return (int(years.group(1)) if years else 0) * 12 + (int(months.group(1)) if months else 0)

def prepare_features(raw):
    x = raw.drop(columns=IDENTIFIERS + ["Credit_Score"], errors="ignore").copy()
    for col in NUMERIC:
        if col in x:
            x[col] = pd.to_numeric(
                x[col].astype("string").str.replace("_", "", regex=False),
                errors="coerce"
            )
    if "Age" in x:
        x.loc[~x["Age"].between(18, 100), "Age"] = np.nan
    for col in ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan"]:
        if col in x:
            x.loc[~x[col].between(0, 20), col] = np.nan
    if "Monthly_Balance" in x:
        x.loc[~x["Monthly_Balance"].between(-10000, 50000), "Monthly_Balance"] = np.nan
    if "Credit_History_Age" in x:
        x["Credit_History_Age"] = x["Credit_History_Age"].map(credit_months)
    if "Type_of_Loan" in x:
        x["Num_of_Loan_Types"] = x["Type_of_Loan"].fillna("").map(
            lambda v: 0 if str(v).strip() in ("", "@", "_") else len(str(v).split(","))
        )
        x = x.drop(columns="Type_of_Loan")
    for col in x.select_dtypes(include=["object", "string"]).columns:
        x[col] = x[col].astype("string").str.replace("_", "", regex=False)
        x[col] = x[col].replace("", pd.NA).astype(object).where(lambda v: v.notna(), np.nan)
    return x
