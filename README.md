# Corporación Favorita — Time Series Sales Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete end‑to‑end time series forecasting project predicting daily unit sales for Corporación Favorita grocery stores. The workflow includes data preparation, feature engineering, model selection, experiment tracking, and deployment through two interactive Streamlit applications.

---

## Live Applications

- **Forecasting App (Deployment)**  
  https://corporacionfavorita-timeseries-forecast.streamlit.app/

- **Model Validation App**  
  https://corporacionfavorita-model-validation.streamlit.app/

---

## Application Overview

### Forecasting App (`app/main.py`)
- Generates single-day or recursive multi‑day forecasts (1–30 days)
- Displays historical sales and future predictions.
- Allows CSV export of forecast results.
- Designed for real‑world deployment conditions where future values are not available.

### Model Validation App (`app/validation_app.py`)
- Compares forecasts vs actuals on the fixed test period (2014‑01‑01 → 2014‑03‑31).
- Computes MAE, RMSE, sMAPE, and daily error analysis.
- Highlights best and worst forecast days.
- Visualizes error distribution and forecast stability.
- Provides model parameters and methodology in the sidebar.

## Presentation
[View Slides (PDF)](docs/SalesForecasting_CorporacionFavorita.pdf)

---

## Project Structure

```
corporacion_favorita/
├── requirements.txt                # Python dependencies
├── README.md
├── LICENSE
│ 
├── app/
│   ├── __init__.py
│   ├── config.py                   # Centralized paths and constants
│   ├── validation_app.py           # Validation app (forecast vs actuals, metrics)
│   └── main.py                     # Deployment app (date picker, pure forecast)
│ 
├── model/
│   ├── __init__.py
│   └── model_utils.py              # Model loading and recursive forecasting
│   └── champion_model.pkl          # Trained XGBoost champion model
│ 
├── data/
│   ├── __init__.py
│   ├── data_utils.py               # Data loading and feature engineering
│   ├── timeseries.csv              # Raw daily sales data
│   ├── timeseries_cleaned.csv      # Cleaned daily sales (output of notebook 01)
│   ├── holidays.csv                # Holiday calendar (National/Regional/Local)
│   ├── oil.csv                     # WTI oil prices
│   ├── stores.csv                  # Store metadata
│   └── results_statistical.csv     # Statistical models evaluation results
│ 
├── assets/
│   └── icon_favicon.png            # Browser favicon
│   └── forecast.png                # Forecast image
│   └── historical.png              # Historical sales image
│ 
└── notebooks/
    ├── 01_favorita_sales_forecasting_data_preparation.ipynb
    ├── 02_statistical_models.ipynb
    └── 03_feature_engineering_models.ipynb
```

---

## Requirements

- Python 3.10+
- macOS / Linux (Windows not tested)
- No GPU required (LSTM excluded from local execution: see note below)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Applications Locally

**Validation app** (forecast vs actuals, metrics):

```bash
streamlit run app/validation_app.py
```

**Forecasting app** (deployment):

```bash
streamlit run app/main.py
```

Open your browser at `http://localhost:8501`

**MLflow UI** (experiment tracking):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser at `http://127.0.0.1:5000`

---

## Forecasting Methodology

1. **Feature engineering**
   
   Lag features (1, 2, 3, 7, 14 days), rolling statistics (7/14/28-day mean and std), cyclical day-of-week encoding (sin/cos), and calendar-based holiday indicators derived from the calendar.
2. **Single-day forecast**
   
   The model predicts the next day using the engineered features computed from the last available date.
3. **Recursive multi-day forecast**
   
   Each prediction is fed back into the feature pipeline to generate the next step, simulating real deployment conditions where future values are not available.
4. **Holiday features**
   
   Holiday effects are reconstructed from `holidays.csv` for any future date, allowing the model to account for National, Regional, and Local holidays beyond the training period.

> **Note on MAE**
> The one-step-ahead MAE reported in the notebooks (95.26) uses real lag values at each step.
> The recursive MAE in the app (~100) uses predicted values as lags, a slightly higher but more realistic estimate of production performance.

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

- LSTM training requires GPU for reasonable speed.

- All other models run efficiently on CPU.

- MLflow tracking is included but optional for running the apps.

---

## Troubleshooting

- **Model not found**  
Make sure the file is in the `model/` folder. Re-run the champion model saving cell in notebook 03.

- **Cleaned dataset missing**  
Re-run notebook 01.

- **MLflow empty**  
Make sure the MLflow UI is stopped before running the logging cells in the notebooks. SQLite does not support concurrent write access.

- **App logo not visible**  
Adjust CSS padding in the Streamlit apps.

---

## Dataset

Based on the Kaggle competition:
[Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)
(aggregated to a single daily time series for this project)

Training period: **2013 only**

Test period: **2014‑01‑01 → 2014‑03‑31**

---

## Author

**Maria Petralia**  
MSIT — Data Science Program  
March 2026
