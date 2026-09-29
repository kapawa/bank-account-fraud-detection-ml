"""
Fraud Detection API — built from Djuichou Kapawa Nounamo's Bank Account Fraud
Detection notebook (IT 7103 Final Project).

WHY THIS FILE IS SIMPLE:
The `rf_pipeline` (or `xgb_pipeline`) is a single sklearn/imblearn Pipeline
object that bundles THREE things together:
    1. preprocessing (imputing, log-transforms, scaling, one-hot encoding)
    2. SMOTE (skipped automatically at prediction time — it only runs during .fit())
    3. the classifier itself

That means this API does NOT need to reimplement any of the preprocessing
logic. It just needs to hand the pipeline a single row of RAW data — the
same 31 raw columns the original `df` had, before any transformation —
and the pipeline does the rest, exactly like it did in the notebook.

HOW TO USE THIS FILE:
1. In the Colab notebook, export the whole fitted pipeline (see the README
   for the exact line to run — it's one line).
2. Move the resulting "model.joblib" file into this same folder.
3. Run locally with:
       uvicorn main:app --reload
4. Open http://127.0.0.1:8000/docs — every field below is pre-filled with a
   realistic default (the median value from the training set), so you can
   hit "Execute" immediately without typing anything.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import sys
import joblib

# ---------------------------------------------------------------------------
# IMPORTANT: the notebook's log_pipeline used two custom functions
# (handle_log_negatives, log_transform) wrapped in FunctionTransformer.
# When joblib saved the pipeline, it didn't save their code — it only saved
# a pointer saying "look for a function with this name in the __main__
# module." In Colab, the notebook's own code IS __main__, so it worked
# there. Here, main.py is a different module, so that pointer fails unless
# we recreate the exact same functions and register them under __main__
# ourselves. This is copied verbatim from the notebook's preprocessing cell.
# ---------------------------------------------------------------------------
def handle_log_negatives(X):
    X_copy = X.copy()
    # Convert values that would make X+0.1 non-positive to NaN so imputer can handle them
    X_copy[X_copy <= -0.1] = np.nan
    return X_copy


def log_transform(X):
    return np.log(X + 0.1)


# Register them under __main__ specifically, since that's where joblib
# expects to find them (matching where they lived in the Colab notebook).
sys.modules["__main__"].handle_log_negatives = handle_log_negatives
sys.modules["__main__"].log_transform = log_transform

# ---------------------------------------------------------------------------
# Load the trained pipeline (preprocessing + classifier bundled together).
# This runs once when the server starts, not on every request.
# ---------------------------------------------------------------------------
try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    model = None  # Lets the app still start even before model.joblib is added

app = FastAPI(
    title="Bank Account Fraud Detection API",
    description="Predicts fraud probability for a bank account application, "
                 "using the Random Forest pipeline from IT 7103 Final Project.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request schema — the 31 raw columns the model was trained on, in the same
# form they appeared in the original dataframe (before preprocessing).
# Defaults below are the MEDIAN value from the training set for each column,
# so the /docs page is testable immediately without guessing realistic values.
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    income: float = Field(default=0.6, description="Annual income, scaled 0.1-0.9")
    name_email_similarity: float = Field(default=0.49)
    prev_address_months_count: int = Field(default=-1, description="-1 means unknown/not available")
    current_address_months_count: int = Field(default=52)
    customer_age: int = Field(default=30)
    days_since_request: float = Field(default=0.0152)
    intended_balcon_amount: float = Field(default=-0.8)
    payment_type: str = Field(default="AA", description="Categorical code, e.g. AA/AB/AC/AD/AE")
    zip_count_4w: int = Field(default=1264)
    velocity_6h: float = Field(default=5300.0)
    velocity_24h: float = Field(default=4746.0)
    velocity_4w: float = Field(default=4910.0)
    bank_branch_count_8w: int = Field(default=9)
    date_of_birth_distinct_emails_4w: int = Field(default=9)
    employment_status: str = Field(default="CA", description="Categorical code, e.g. CA/CB/CC...")
    credit_risk_score: int = Field(default=122)
    email_is_free: int = Field(default=1, description="1 = free email provider, 0 = not")
    housing_status: str = Field(default="BA", description="Categorical code, e.g. BA/BB/BC...")
    phone_home_valid: int = Field(default=0)
    phone_mobile_valid: int = Field(default=1)
    bank_months_count: int = Field(default=5)
    has_other_cards: int = Field(default=0)
    proposed_credit_limit: float = Field(default=200.0)
    foreign_request: int = Field(default=0)
    source: str = Field(default="INTERNET", description="INTERNET or TELEAPP")
    session_length_in_minutes: float = Field(default=5.08)
    device_os: str = Field(default="windows", description="windows/linux/macintosh/x11/other")
    keep_alive_session: int = Field(default=1)
    device_distinct_emails_8w: int = Field(default=1)
    device_fraud_count: int = Field(default=0)
    month: int = Field(default=3)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fraud detection API is running. Visit /docs to try it."}


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        return {"error": "Model not loaded yet. Add model.joblib to this folder and restart."}

    # Build a single-row DataFrame with the exact column names the
    # ColumnTransformer expects. Column ORDER doesn't matter — the
    # ColumnTransformer selects columns by name, not position.
    row = pd.DataFrame([transaction.model_dump()])

    # The pipeline handles imputing, log-transforms, scaling and one-hot
    # encoding internally — SMOTE is automatically skipped at prediction time.
    probability = float(model.predict_proba(row)[0][1])
    is_fraud = probability >= 0.5

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": is_fraud,
    }
