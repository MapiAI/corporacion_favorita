"""
model_utils.py — Model loading and recursive forecasting for Favorita Sales Forecasting.

This module provides:
- load_model()         : loads the champion XGBoost model from .pkl
- recursive_forecast() : autoregressive multi-day forecast
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# Model loading ---------------------------------------

def load_model(model_path: Path):
    """
    Load the champion XGBoost model from a joblib pickle file.

    Parameters -------------
    model_path : Path
        Full path to champion_model.pkl.

    Returns --------------
    XGBRegressor
        Trained XGBoost champion model.

    Raises ----------------
    FileNotFoundError
        If champion_model.pkl is not found at the given path.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at {model_path}. "
            "Re-run the model saving cell in notebook 03."
        )
    return joblib.load(model_path)


# Recursive forecasting ---------------------------------------
def recursive_forecast(
    model,
    history: pd.Series,
    n_days: int,
    feature_cols: list,
    build_features_fn,
    get_holiday_features_fn,
    holiday_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Autoregressive recursive forecast for n_days starting the day after
    the last date in history.

    Each predicted value feeds back as input for the next step,
    simulating real deployment conditions where future values are unknown.

    Parameters -------------
    model : XGBRegressor
        Trained champion model.
    history : pd.Series
        Daily unit_sales series with DatetimeIndex up to the cutoff date.
    n_days : int
        Number of days to forecast.
    feature_cols : list
        Ordered list of feature column names (must match training order).
    build_features_fn : callable
        Feature engineering function from data_utils.
    get_holiday_features_fn : callable
        Holiday feature lookup function from data_utils.
    holiday_features : pd.DataFrame
        Prebuilt holiday feature table from data_utils.build_holiday_features().

    Returns ---------------------
    pd.DataFrame
        DataFrame with DatetimeIndex and a single 'forecast' column.
    """
    history   = history.copy()
    last_date = history.index[-1]
    predictions = []

    for _ in range(n_days):
        next_date = last_date + pd.Timedelta(days=1)

        # Build features from current history window
        feat_df  = build_features_fn(history)
        last_row = feat_df.iloc[[-1]].copy()
        last_row.index = [next_date]

        # Overwrite cyclical features for the actual next date
        dow = next_date.dayofweek
        last_row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        last_row["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        # Add holiday features for the next date
        hol = get_holiday_features_fn(next_date, holiday_features)
        for col, val in hol.items():
            last_row[col] = val

        # Predict and clip to non-negative
        X      = last_row[feature_cols]
        y_pred = max(0.0, float(model.predict(X)[0]))

        predictions.append({"date": next_date, "forecast": round(y_pred, 2)})

        # Feed prediction back into history for the next iteration
        history.loc[next_date] = y_pred
        last_date = next_date

    return pd.DataFrame(predictions).set_index("date")