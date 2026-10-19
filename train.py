"""Train a reproducible educational credit-score classifier.

Usage: python train.py --data data/train.csv
Raw data and trained artifacts are deliberately excluded from Git.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from model import INTERFACE, prepare_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional quick smoke test (not for reporting final metrics)")
    args = parser.parse_args()
    raw = pd.read_csv(args.data, low_memory=False)
    raw = raw.loc[raw["Credit_Score"].isin(["Poor", "Standard", "Good"])].copy()
    if args.limit:
        raw = raw.head(args.limit).copy()
    labels = {"Poor": 0, "Standard": 1, "Good": 2}
    y = raw["Credit_Score"].map(labels)
    X = prepare_features(raw)
    # Same customer appears across months; keep customers out of both splits.
    groups = raw["Customer_ID"].astype(str)
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    ).split(X, y, groups=groups))
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    num_cols = x_train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in x_train if c not in num_cols]
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), num_cols),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    pipeline = Pipeline([
        ("prepare", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=80, max_depth=12, random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "test_rows": int(len(test_idx)),
        "train_rows": int(len(train_idx)),
        "classification_report": classification_report(
            y_test, predictions, labels=[0,1,2],
            target_names=["Poor", "Standard", "Good"], output_dict=True,
            zero_division=0
        )
    }
    # For the seven-input UI, other features are filled with TRAINING-ONLY
    # medians or modes, then processed by the SAME fitted pipeline.
    defaults = {}
    for col in X.columns:
        if col in num_cols:
            val = x_train[col].median()
            defaults[col] = float(val) if pd.notna(val) else 0.0
        else:
            mode = x_train[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
    bundle = {"pipeline": pipeline, "defaults": defaults,
              "columns": X.columns.tolist(), "inputs": INTERFACE}
    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    joblib.dump(bundle, out/"credit_score_model.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Group-held-out accuracy: {metrics['accuracy']:.3f}")
    print(f"Saved {out/'credit_score_model.joblib'}")

if __name__ == "__main__":
    main()
