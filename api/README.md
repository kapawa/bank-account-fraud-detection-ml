# Fraud Detection API — Step-by-Step Guide
### Built from my IT 7103 Bank Account Fraud Detection notebook

Follow the steps in order if you want to reproduce this work.

## Part 1: Export your trained model from Colab

The notebook already found the best model — Random Forest, selected in my own conclusion for having the highest F1-score (0.209). That object lives in the notebook as:
```python
best_models['Random Forest'].best_estimator_
```
This is the FULL pipeline: preprocessing + SMOTE + classifier, all bundled together. You don't need to extract or reimplement any of the preprocessing steps — saving this one object saves everything.

### 1. Add a new cell at the END of your Colab notebook and run it:
```python
import joblib
final_model = best_models['Random Forest'].best_estimator_
joblib.dump(final_model, '/content/drive/MyDrive/Practical_data_analytics/Final_Project/model.joblib')
```
(If you'd rather deploy XGBoost instead — F1 was 0.1967, close behind — swap the key to `best_models['XGBoost']`.)

### 2. Download the file to your own laptop
Since Colab runs on Google's servers, not your computer, you need to bring the file down locally. Easiest way — add and run:
```python
from google.colab import files
files.download('/content/drive/MyDrive/Practical_data_analytics/Final_Project/model.joblib')
```
This will prompt a normal browser download. Save it, then move `model.joblib` into this same `fraud-api` folder (next to `main.py`).

## Part 2: Get it running on your own laptop

### 3. Install the tools (one time only)
Open a terminal in this folder and run:
```
pip install -r requirements.txt
```

### 4. Run the server
```
uvicorn main:app --reload
```
You should see it running on `http://127.0.0.1:8000`.

### 5. Test it
Open your browser to:
```
http://127.0.0.1:8000/docs
```
Click on `/predict`, then "Try it out." Every field is already pre-filled with the **median value from your own training set** — so you can just click "Execute" immediately and see a real fraud probability come back, no guessing required.

**Try this to see it actually working (not just running):** change `credit_risk_score` to something like `-100` and execute again. Credit risk score was one of your top predictive features (per your Hypothesis 1 analysis) — watch the fraud probability move.

## Part 3: Put it on the actual internet

### 6. Push this folder to GitHub
Create a new repo (e.g. `fraud-detection-api`) and push `main.py`, `requirements.txt`, and `model.joblib`.

> Note: `model.joblib` may be a few MB. If GitHub rejects it for size, Render also lets you upload it directly through their dashboard instead of via git — ask if you hit this.

### 7. Deploy on Render (free tier, no credit card needed)
1. Go to render.com and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy. Takes a few minutes.

### 8. You now have a live link
Something like `https://fraud-detection-api-xxxx.onrender.com/docs`. Test it from your phone to confirm it's really live. Add this link to your resume next to the fraud detection project, and to your GitHub README.

## If something breaks
- **Model not found** → double-check `model.joblib` is in this same folder, spelled exactly that way.
- **"No module named imblearn"** → run `pip install imbalanced-learn` (the package name differs from the import name).
- **A category you type isn't recognized** → not a problem. Your pipeline uses `OneHotEncoder(handle_unknown='ignore')`, so any category the model hasn't seen before is safely ignored rather than causing an error.
- **Fraud probability always near 0** → expected. Your notebook shows fraud is rare (~1% of the data) and precision/recall are both modest (F1 ≈ 0.21) — this is a known limitation you already documented in your own conclusion, not a bug in the API.

You don't need this to be perfect. A working `/predict` endpoint that returns a probability is the whole goal — everything else is polish.
