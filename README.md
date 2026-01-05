# Bank Account Fraud Detection using Machine Learning

## Overview
This project implements an end-to-end supervised machine learning pipeline to detect fraudulent bank accounts using a highly imbalanced dataset (~1% fraud). The objective is to maximize fraud detection while controlling false positives.

## Dataset
- NeurIPS 2022 Bank Account Fraud Dataset
- Highly imbalanced binary classification problem
- Raw data not included due to licensing restrictions

## Approach
- Data cleaning and feature engineering
- Log transformations, one-hot encoding, and standardization
- SMOTE to address class imbalance
- Model training and hyperparameter tuning with GridSearchCV

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Results
- Achieved ~200% improvement in F1-score compared to baseline Logistic Regression
- Ensemble models significantly outperformed linear models on imbalanced data
- Evaluation metrics: F1-score, precision, recall, ROC-AUC

## Tools & Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn

## How to Run
Install dependencies:
```bash
pip install -r requirements.txt
