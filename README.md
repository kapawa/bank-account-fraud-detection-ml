# Bank Account Fraud Detection using Machine Learning

## 🔴 Live Demo
The trained model is deployed as a live REST API:
**[fraud-detection-api-39uj.onrender.com/docs](https://fraud-detection-api-39uj.onrender.com/docs)**

Try it directly in the interactive docs — every field is pre-filled with realistic
default values from the training set, so you can hit "Execute" immediately.
Built with FastAPI, deployed on Render (free tier spins down after inactivity,
so the first request may take 30–50 seconds to wake up).

## Overview
This project implements an end-to-end supervised machine learning pipeline to detect fraudulent bank accounts using a highly imbalanced dataset (~1% fraud). The objective is to maximize fraud detection while controlling false positives.

## Dataset
- NeurIPS 2022 Bank Account Fraud Dataset
- Highly imbalanced binary classification problem (~1% fraud, 99% legitimate)
- Raw data not included due to licensing restrictions

## Approach
- Data cleaning and feature engineering
- Log transformations, one-hot encoding, and standardization
- SMOTE to address class imbalance
- Model training and hyperparameter tuning with GridSearchCV (3-fold CV, F1-optimized)

## Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Results
| Model | F1-Score (Fraud) | Recall (Fraud) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.066 | 0.565 | 0.815 |
| **Random Forest (best)** | **0.209** | 0.304 | 0.797 |
| XGBoost | 0.197 | 0.261 | 0.818 |

- Random Forest achieved the highest F1-score — a ~215% improvement over the Logistic Regression baseline
- Logistic Regression had the highest recall, illustrating the precision/recall trade-off inherent to fraud detection
- Feature importance analysis confirmed behavioral/financial features (customer age, payment type, credit risk score) are strong fraud predictors

**Limitations:** Absolute F1-scores remain modest, reflecting the genuine difficulty of fraud detection under severe class imbalance (92 fraud cases out of 9,600 training records) and a 1.2% data subset used for computational efficiency. In production, further gains would likely come from more training data, cost-sensitive learning, or anomaly-detection approaches.

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, FastAPI, Render

## How to Run

**1. Explore the analysis:**
```bash
pip install -r requirements.txt
jupyter notebook Final_Fraud_Detection_Notebook.ipynb
```

**2. Run the API locally:**
```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```
Then visit `http://127.0.0.1:8000/docs`.