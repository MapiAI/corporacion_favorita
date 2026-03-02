# Corporación Favorita — Sales Forecasting

A time series forecasting project that predicts daily unit sales for Corporación Favorita grocery stores using classical statistical models, machine learning, and deep learning, deployed as an interactive Streamlit app.

---

## What the Apps Do

**Deployment app (`app/main.py`)**
- Select a forecast cut-off date and forecast mode (single day or next N days)
- View the last 6 months of sales history overlaid with the forecast
- Download the forecast as a CSV file

**Validation app (`app/validation_app.py`)**
- Fixed forecast from 2014-01-01 over a configurable horizon (7–90 days)
- Switch between deployment mode (forecast only) and validation mode (forecast vs actuals)
- View MAE, RMSE, sMAPE metrics and best/worst forecast day
- Explore model parameters and methodology directly in the sidebar

---

## Project Structure

```
corporacion_favorita/
├── champion_model.pkl              # Trained XGBoost champion model
├── requirements.txt                # Python dependencies
├── requirements_streamlit.txt      # Streamlit-specific dependencies
├── mlflow.db                       # MLflow experiment tracking (SQLite)
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py                   # Centralized paths and constants
│   ├── app.py                      # Validation app (forecast vs actuals, metrics)
│   └── main.py                     # Deployment app (date picker, pure forecast)
├── model/
│   ├── __init__.py
│   └── model_utils.py              # Model loading and recursive forecasting
├── data/
│   ├── __init__.py
│   ├── data_utils.py               # Data loading and feature engineering
│   ├── timeseries.csv              # Raw daily sales data
│   ├── timeseries_cleaned.csv      # Cleaned daily sales (output of notebook 01)
│   ├── holidays.csv                # Holiday calendar (National/Regional/Local)
│   ├── oil.csv                     # WTI oil prices
│   ├── stores.csv                  # Store metadata
│   └── results_statistical.csv     # Statistical models evaluation results
├── assets/
│   └── icon_favicon.png            # Browser favicon
└── notebooks/
    ├── 01_favorita_sales_forecasting_data_preparation.ipynb
    ├── 02_statistical_models.ipynb
    └── 03_feature_engineering_models.ipynb
```

---

## Requirements

- Python 3.10
- macOS / Linux (Windows not tested)
- No GPU required (LSTM excluded from local execution: see note below)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Instructions

**Validation app** (forecast vs actuals, metrics):

```bash
conda activate timeseries
cd /path/to/corporacion_favorita
streamlit run app/validation_app.py
```

**Deployment app** (date picker, pure forecast):

```bash
conda activate timeseries
cd /path/to/corporacion_favorita
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`

**MLflow UI** (experiment tracking):

```bash
conda activate timeseries
cd /path/to/corporacion_favorita
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser at `http://127.0.0.1:5000`

---

## How Predictions Work

1. **Feature engineering** — lag features (1, 2, 3, 7, 14 days), rolling statistics (7/14/28-day mean and std), cyclical day-of-week encoding (sin/cos), and calendar-based holiday indicators are computed from the sales history.
2. **Single-day forecast** — the model predicts the next day using the engineered features from the last known date.
3. **Recursive multi-day forecast** — each prediction feeds back as input for the next step, simulating real deployment conditions where future values are never available in advance.
4. **Holiday features** — reconstructed from `holidays.csv` for any future date, allowing the model to account for National, Regional, and Local holidays beyond the training period.

> **Note on MAE**: the one-step-ahead MAE reported in the notebooks (95.26) uses real lag values at each step. The recursive MAE in the app (~101) uses predicted values as lags — a slightly higher but more realistic estimate of production performance.

---

## Models Evaluated

| Model | MAE | RMSE | sMAPE | Family |
|---|---|---|---|---|
| **XGBoost (HyperOpt tuned)** ✓ | **95.26** | **140.39** | **19.95%** | ML |
| XGBoost baseline | 95.52 | 140.94 | 20.04% | ML |
| SARIMAX | 96.66 | 143.01 | 20.38% | Statistical |
| Random Forest (tuned) | 96.70 | 141.77 | 20.15% | ML |
| Holt-Winters | 97.44 | 143.86 | 20.57% | Statistical |
| Random Forest | 97.42 | 140.80 | 20.33% | ML |
| SARIMA | 97.80 | 145.23 | 20.60% | Statistical |
| Linear Regression | 102.58 | 151.33 | 21.36% | ML |
| LSTM | 108.01 | 161.40 | 22.33% | DL |
| Prophet | 109.95 | 152.26 | 23.45% | Statistical |

All models evaluated on the same fixed test period: **2014-01-01 → 2014-03-31**.  
Primary metric: **MAE** (interpretable, unit-consistent, applicable across all model families).

---

## Computational Note

The LSTM model in notebook 03 is computationally intensive and may be slow to train on hardware without GPU support. All other models run efficiently on standard hardware. LSTM results were obtained on GPU hardware and are reported manually in the notebook.

---

## Troubleshooting

**`champion_model.pkl` not found**  
Make sure the file is in the `model/` folder. Re-run the champion model saving cell in notebook 03.

**`timeseries_cleaned.csv` not found**  
Re-run notebook 01 to regenerate the cleaned dataset in `data/`.

**MLflow shows no experiments**  
Make sure the MLflow UI is stopped before running the logging cells in the notebooks. SQLite does not support concurrent write access.

**App logo not visible**  
Increase `padding-top` in the CSS block at the top of `app/validation_app.py` or `app/main.py`.

---

## Dataset

Based on the [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) Kaggle competition (simplified version).  
Training data covers **2013 only** (single aggregated daily time series, 454 observations).

---

## Author

**Maria Petralia**  
MSIT — Data Science Program  
March 2026
