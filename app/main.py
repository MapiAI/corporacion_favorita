import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64
from pathlib import Path

# Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Page config
st.set_page_config(
    page_title="Favorita Sales Forecasting",
    page_icon="assets/icon_favicon.png",
    layout="wide"
)

# Global CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.3rem;
    }
    hr {
        margin: 0.3rem 0;
    }
    div.stButton > button[kind="primary"] {
        background-color: #c0392b;
        border-color: #c0392b;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #a93226;
        border-color: #a93226;
    }
    </style>
""", unsafe_allow_html=True)

# Paths 
from app.config import (
    BASE_DIR, DATA_PATH, MODEL_PATH, ASSETS_PATH,
    FEATURE_COLS, TRAIN_END, TEST_START,
    HISTORY_WINDOW_DAYS,
    FORECAST_MIN_MAIN, FORECAST_MAX_MAIN  
)
from data.data_utils import (
    load_data, load_holidays,
    build_holiday_features, get_holiday_features,
    build_features
)
from model.model_utils import load_model, recursive_forecast

# Helper: image to base64
def img_to_base64(path: Path) -> str:
    """Convert a local image file to a base64 string for inline HTML embedding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load data and model 
@st.cache_data
def _load_data():
    return load_data(DATA_PATH)

@st.cache_data
def _load_holidays():
    return load_holidays(DATA_PATH)

@st.cache_resource
def _load_model():
    return load_model(MODEL_PATH)

df       = _load_data()
holidays = _load_holidays()
model    = _load_model()

# Holiday lookup 
holiday_features = build_holiday_features(holidays)

# Header
icon_b64 = img_to_base64(ASSETS_PATH / "icon_favicon.png")
st.markdown(f"""
    <div style='display:flex; align-items:center; gap:14px; padding:8px 0 16px 0;'>
        <img src='data:image/png;base64,{icon_b64}' width='44'>
        <span style='font-size:26px; font-weight:600; color:#3d3d3d;'>Corporación Favorita</span>
        <span style='font-size:26px; font-weight:300; color:#888;'>Sales Forecasting</span>
    </div>
""", unsafe_allow_html=True)
st.caption("🚀 Deployment Mode — forecast from any date")
st.divider()

# Sidebar 
st.sidebar.markdown("**Mode:** 🚀 Deployment")
st.sidebar.header("Forecast Settings")

# Date picker — user selects the cutoff date
cutoff_date = st.sidebar.date_input(
    "Forecast cut-off date",
    value=pd.Timestamp("2013-12-31").date(),
    min_value=df.index.min().date(),
    max_value=df.index.max().date(),
    help="The model uses history up to this date. Forecast starts the following day."
)
if cutoff_date is None:
    st.warning("Please select a valid date to continue.")
    st.stop()

cutoff_ts = pd.Timestamp(cutoff_date)

# Forecast mode
mode = st.sidebar.radio(
    "Forecast mode",
    ["Single day", "Next N days"],
    horizontal=False
)
n_days = st.sidebar.slider(
    "N days",
    min_value=1,
    max_value=30,
    value=7,
    help="Number of days to forecast ahead. Longer horizons accumulate more uncertainty. Each prediction feeds the next step."
) if mode == "Next N days" else 1

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost (HyperOpt tuned)")
st.sidebar.markdown("**Trained on:** 2013-01-02 → 2013-12-31")
st.sidebar.markdown("**Forecast method:** Recursive")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Parameters**")
st.sidebar.markdown("- `max_depth` = 2")
st.sidebar.markdown("- `n_estimators` = 50")
st.sidebar.markdown("- `learning_rate` = 0.077")
st.sidebar.markdown("- `subsample` = 0.711")
st.sidebar.markdown("- `colsample_bytree` = 0.939")
st.sidebar.markdown("- `min_child_weight` = 2")

# Historical data 
hist_icon_b64 = img_to_base64(ASSETS_PATH / "historical.png")
st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
        <img src='data:image/png;base64,{hist_icon_b64}' height='32'>
        <span style='font-size:28px; font-weight:600; color:#3d3d3d;'>Historical Sales</span>
    </div>
""", unsafe_allow_html=True)

# Show last 6 months of history up to the cutoff date
history_window = df.loc[:cutoff_ts]["unit_sales"].last("180D")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(history_window.index, history_window.values, linewidth=1, color="steelblue")
ax.axvline(cutoff_ts, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Forecast start")
ax.set_xlabel("Date")
ax.set_ylabel("Unit Sales")
ax.set_title(f"Daily Unit Sales — Last 6 Months up to {cutoff_date}")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
st.divider()

# Forecast section 
forecast_label = "next day" if mode == "Single day" else f"next {n_days} days"
# Forecast section header
fore_icon_b64 = img_to_base64(ASSETS_PATH / "forecast.png")
st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
        <img src='data:image/png;base64,{fore_icon_b64}' height='32'>
        <span style='font-size:28px; font-weight:600; color:#3d3d3d;'>Forecast</span>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    f"Forecasting **{forecast_label}** from **{cutoff_date}**."
)

# Expander 
with st.expander("ℹ️ How does this forecast work?"):
    st.markdown("""
    **Recursive Forecasting with XGBoost**

    This app uses a recursive (autoregressive) forecasting strategy:

    1. **Day 1** — the model uses the last known sales values as input features
        (lags, rolling averages) and predicts the next day.
    2. **Day 2** — since the actual value for Day 1 is unknown, the prediction
        from Step 1 is used as input to predict Day 2.
    3. **Day N** — each prediction feeds back into the feature pipeline to produce
        the next forecast.

    This simulates real deployment conditions where future values are never
    available in advance.

    **Why does error increase with the horizon?**
    Each prediction carries a small error. When that prediction is used as input
    for the next step, the error propagates forward. Longer horizons accumulate
    more uncertainty — this is an inherent characteristic of recursive forecasting.
    """)

# Run Forecast button 
if st.button("▶ Run Forecast", type="primary"):

    # Validate: enough history before cutoff
    history_series = df.loc[:cutoff_ts]["unit_sales"]
    if len(history_series) < 30:
        st.error("Not enough history before the selected date. Please choose a later date.")
        st.stop()

    with st.spinner("Running recursive forecast..."):
        forecast_df = recursive_forecast(
            model, history_series, n_days,
            feature_cols=FEATURE_COLS,
            build_features_fn=build_features,
            get_holiday_features_fn=get_holiday_features,
            holiday_features=holiday_features
        )

    # Plot history + forecast 
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    hist_plot = history_series.last("180D")
    ax2.plot(hist_plot.index, hist_plot.values,
            label="Sales history", color="steelblue", linewidth=1.5)
    ax2.plot(forecast_df.index, forecast_df["forecast"].values,
            label=f"XGBoost forecast ({n_days} day{'s' if n_days > 1 else ''})",
            color="tomato", linewidth=2, linestyle="--", marker="o", markersize=4)
    ax2.axvline(cutoff_ts, color="gray",
                linestyle=":", alpha=0.7, label="Forecast start")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Unit Sales")
    ax2.set_title(f"XGBoost Recursive Forecast — {n_days} Day{'s' if n_days > 1 else ''} from {cutoff_date}")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # Single day result
    if mode == "Single day":
        predicted_value = forecast_df["forecast"].iloc[0]
        predicted_date  = forecast_df.index[0].strftime("%Y-%m-%d")
        st.markdown(f"""
            <div style='display:flex; justify-content:center; margin:16px 0;'>
                <div style='border:1px solid #ddd; border-radius:6px; padding:20px 40px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>Predicted Sales for {predicted_date}</div>
                    <div style='font-size:32px; font-weight:600; color:#c0392b;'>{predicted_value:.0f}</div>
                    <div style='font-size:12px; color:#aaa;'>units</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Forecast table 
    st.markdown("#### Forecast Table")
    comparison = forecast_df.copy()
    comparison = comparison.rename(columns={"forecast": "Forecast"})
    comparison.index = comparison.index.strftime("%Y-%m-%d")
    st.dataframe(comparison, use_container_width=True)

    # CSV download 
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=comparison.to_csv(),
        file_name=f"favorita_forecast_{n_days}d_from_{cutoff_date}.csv",
        mime="text/csv"
    )