# Credit Score Classification — MSc Portfolio Prototype

A small, educational machine-learning web app adapted from an MSc Predictive
Analytics dissertation. It uses a **Random Forest** to predict the dataset's
**Poor / Standard / Good credit-score category** from seven user-controlled
inputs and training-set defaults for other features.

**It does not determine bank loan eligibility or approval.**

## Features
- Python + scikit-learn training pipeline and Gradio interface
- Explicit numeric cleaning, categorical encoding and missing-value handling
- Train/test split by **Customer_ID**, reducing leakage across repeated monthly
  records from the same customer
- Model serialization; the app does not retrain on every launch
- No applicant names or ID numbers in the user interface
- Basic input validation and clear model limitations

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
mkdir -p data
# Put your legitimately obtained original train.csv in data/train.csv
python train.py --data data/train.csv
python app.py
```

Training writes `artifacts/credit_score_model.joblib` and `metrics.json`.
Both `data/` and `artifacts/` are ignored by Git. Only load `.joblib` files
you generated or otherwise trust.

## Data and privacy
The original academic dataset includes identifier-like columns (such as
names, customer IDs and SSNs). These are **dropped from model features**;
Customer_ID is used only to group the train/test split. **Do not commit or
redistribute the original dataset without checking its licence and privacy
implications.** A dataset download/source link can be added once verified.

## Limitations and next steps
- This is an educational credit-score **classification**, not real credit
  scoring, a loan eligibility assessment, or a regulated decision system.
- Seven visible inputs do not fully describe a customer; the remaining
  features are held at training-set defaults.
- Training metrics should be reported only after running the full dataset.
- This project uses **Python and Gradio**, not JavaScript/TypeScript or React.
  A future iteration could use a JS frontend and Python prediction API.
- It is a **portfolio adaptation**, not a claim of a deployed production system.

## Origins
Adapted from the author's MSc Predictive Analytics dissertation notebook.
The portfolio version simplifies the original model comparison into a
reproducible, standalone Random Forest demo and improves the disclosure
of prediction scope.
