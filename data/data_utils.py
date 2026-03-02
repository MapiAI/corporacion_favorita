"""
data_utils.py — Data loading and feature engineering for Favorita Sales Forecasting.

This module provides:
- load_data()         : loads the cleaned time series and holidays CSV
- build_features()    : engineers lag, rolling, cyclical and holiday features
- get_holiday_features(): returns holiday flags for any given date
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Data loading ------------------------------------------
def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the cleaned daily time series from timeseries_cleaned.csv.

    Parameters
    ----------
    data_path : Path
        Path to the data/ folder.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and unit_sales column,
        enforced to daily frequency.
    """
    df = pd.read_csv(data_path / "timeseries_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.asfreq("D")
    return df


def load_holidays(data_path: Path) -> pd.DataFrame:
    """
    Load the holidays dataset from holidays.csv.

    Parameters
    ----------
    data_path : Path
        Path to the data/ folder.

    Returns
    -------
    pd.DataFrame
        DataFrame with date, locale, locale_name, description columns.
    """
    df = pd.read_csv(data_path / "holidays.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# Holiday feature lookup ------------------------------------------
def build_holiday_features(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily holiday feature table from the holidays DataFrame.

    Creates binary indicators for:
    - is_holiday     : any holiday type
    - is_national    : National holiday
    - is_regional    : Regional holiday
    - is_local       : Local holiday
    - is_weekend     : Saturday or Sunday
    - holiday_weekday: holiday occurring on a weekday
    - holiday_weekend: holiday occurring on a weekend

    Parameters
    ----------
    holidays_df : pd.DataFrame
        Raw holidays DataFrame loaded from holidays.csv.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame indexed by date with holiday feature columns.
    """
    holiday_dates  = holidays_df["date"].unique()
    national_dates = holidays_df[holidays_df["locale"] == "National"]["date"].unique()
    regional_dates = holidays_df[holidays_df["locale"] == "Regional"]["date"].unique()
    local_dates    = holidays_df[holidays_df["locale"] == "Local"]["date"].unique()

    all_dates = pd.date_range(holidays_df["date"].min(), holidays_df["date"].max(), freq="D")
    hol = pd.DataFrame(index=all_dates)

    hol["is_holiday"]  = hol.index.isin(holiday_dates).astype(int)
    hol["is_national"] = hol.index.isin(national_dates).astype(int)
    hol["is_regional"] = hol.index.isin(regional_dates).astype(int)
    hol["is_local"]    = hol.index.isin(local_dates).astype(int)

    hol["is_weekend"]      = (hol.index.dayofweek >= 5).astype(int)
    hol["holiday_weekday"] = hol["is_holiday"] * (1 - hol["is_weekend"])
    hol["holiday_weekend"] = hol["is_holiday"] * hol["is_weekend"]

    return hol


def get_holiday_features(date: pd.Timestamp, holiday_features: pd.DataFrame) -> dict:
    """
    Return holiday feature flags for a given date.

    If the date is outside the holiday lookup table range, returns zeros.

    Parameters
    ----------
    date : pd.Timestamp
        The date to look up.
    holiday_features : pd.DataFrame
        Prebuilt holiday feature table from build_holiday_features().

    Returns
    -------
    dict
        Dictionary with keys: holiday_weekday, holiday_weekend,
        is_national, is_regional, is_local.
    """
    if date in holiday_features.index:
        row = holiday_features.loc[date]
        return {
            "holiday_weekday": int(row["holiday_weekday"]),
            "holiday_weekend": int(row["holiday_weekend"]),
            "is_national":     int(row["is_national"]),
            "is_regional":     int(row["is_regional"]),
            "is_local":        int(row["is_local"]),
        }
    return {
        "holiday_weekday": 0,
        "holiday_weekend": 0,
        "is_national":     0,
        "is_regional":     0,
        "is_local":        0,
    }


# Feature engineering ------------------------------------------
def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Engineer lag, rolling, and cyclical features from a daily sales series.

    Features created:
    - lag_1, lag_2, lag_3, lag_7, lag_14  : lagged unit sales
    - roll_mean_7, roll_mean_14, roll_mean_28 : rolling means (shift=1 to avoid leakage)
    - roll_std_7   : rolling standard deviation (shift=1)
    - dow_sin, dow_cos : cyclical day-of-week encoding

    Parameters
    ----------
    series : pd.Series
        Daily unit_sales series with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with unit_sales and all engineered feature columns.
    """
    df_feat = pd.DataFrame({"unit_sales": series})

    # Lag features — capture short and medium-term demand patterns
    for l in [1, 2, 3, 7, 14]:
        df_feat[f"lag_{l}"] = df_feat["unit_sales"].shift(l)

    # Rolling statistics — shift(1) prevents data leakage into the current day
    df_feat["roll_mean_7"]  = df_feat["unit_sales"].shift(1).rolling(7).mean()
    df_feat["roll_mean_14"] = df_feat["unit_sales"].shift(1).rolling(14).mean()
    df_feat["roll_mean_28"] = df_feat["unit_sales"].shift(1).rolling(28).mean()
    df_feat["roll_std_7"]   = df_feat["unit_sales"].shift(1).rolling(7).std()

    # Cyclical day-of-week encoding — preserves circular structure (Mon-Sun proximity)
    dow = df_feat.index.dayofweek
    df_feat["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df_feat