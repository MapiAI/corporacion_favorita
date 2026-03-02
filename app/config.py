"""
config.py — Centralized configuration for Favorita Sales Forecasting.

All paths, constants, and model parameters are defined here so that
app.py and main.py stay clean and easy to maintain.

Usage:
    from app.config import BASE_DIR, DATA_PATH, MODEL_PATH, FEATURE_COLS
"""

from pathlib import Path

# Project root ---------------------------------------
# __file__ is app/config.py → .parent is app/ → .parent is corporacion_favorita/
BASE_DIR    = Path(__file__).resolve().parent.parent
APP_DIR     = Path(__file__).resolve().parent

# Data and model paths -----------------------------------
DATA_PATH   = BASE_DIR / "data"
MODEL_PATH  = BASE_DIR / "model" / "champion_model.pkl"
ASSETS_PATH = BASE_DIR / "assets"

# MLflow --------------------------------------------------
# SQLite backend — avoids the filesystem deprecation warning in MLflow 3.x.
# Launch UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR}/mlflow.db"
EXPERIMENT_NAME     = "Favorita_Sales_Forecasting"

# Training and test periods -------------------------------
TRAIN_START = "2013-01-02"
TRAIN_END   = "2013-12-31"
TEST_START  = "2014-01-01"
TEST_END    = "2014-03-31"

# Feature columns (must match training order exactly) -----------------    
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "dow_sin", "dow_cos",
    "holiday_weekday", "holiday_weekend",
    "is_national", "is_regional", "is_local"
]

# Champion model parameters (XGBoost HyperOpt tuned) ------------------------------
# Selected based on lowest MAE (95.26) on fixed test period 2014-01-01 → 2014-03-31.
CHAMPION_PARAMS = {
    "max_depth":        2,
    "n_estimators":     50,
    "learning_rate":    0.077,
    "subsample":        0.711,
    "colsample_bytree": 0.939,
    "min_child_weight": 2,
    "random_state":     3,
}

# App display settings ---------------------------------------
HISTORY_WINDOW_DAYS = 180 # days of history shown in plots (6 months)

# app.py (validation app) slider settings
FORECAST_MIN_APP = 7
FORECAST_MAX_APP = 90

# main.py (deployment app) slider settings
FORECAST_MIN_MAIN = 1
FORECAST_MAX_MAIN = 30
