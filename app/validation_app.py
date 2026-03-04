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
    FORECAST_MIN_APP, FORECAST_MAX_APP
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

# Split training and test periods
df_train = df.loc[:TRAIN_END]
df_test  = df.loc[TEST_START:]

# Build holiday lookup 
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
st.caption("🔬 Validation Mode — forecast evaluated against actual sales")
st.divider()

# Sidebar
st.sidebar.markdown("**Mode:** 🔬 Validation")
st.sidebar.header("Forecast Settings")
n_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    help="Number of days to forecast. The model was evaluated on 90 days (Jan 1 → Mar 31 2014)."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost (HyperOpt tuned)")
st.sidebar.markdown("**Trained on:** 2013-01-02 → 2013-12-31")
st.sidebar.markdown("**Test period:** 2014-01-01 → 2014-03-31")
st.sidebar.markdown("**Test MAE (one-step):** 95.26 units/day")
st.sidebar.markdown("**Forecast method:** Recursive")
st.sidebar.markdown("---")
show_actuals = st.sidebar.checkbox("Show actual values (validation mode)", value=True)
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
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_train.index, df_train["unit_sales"], linewidth=1, color="steelblue")
ax.set_xlabel("Date")
ax.set_ylabel("Unit Sales")
ax.set_title("Daily Unit Sales — Training Period (2013)")
plt.tight_layout()
st.pyplot(fig)
st.divider()

# Forecast section
forecast_start = df_train.index[-1] + pd.Timedelta(days=1)
fore_icon_b64 = img_to_base64(ASSETS_PATH / "forecast.png")
st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
        <img src='data:image/png;base64,{fore_icon_b64}' height='32'>
        <span style='font-size:28px; font-weight:600; color:#3d3d3d;'>Forecast</span>
    </div>
""", unsafe_allow_html=True)
st.markdown(
    f"Forecasting **{n_days} days** from **{forecast_start.date()}**. "
    + ("Actual values (test set) are shown for comparison."
    if show_actuals else
    "Running in deployment mode — no actual values shown.")
)

# ── Expander ──────────────────────────────────────────────────────────────────
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
    for the next step, the error propagates forward. This is why MAE is slightly
    higher in the app (~101) compared to the one-step-ahead evaluation in the
    notebook (~95).
    """)

# ── Run Forecast button ───────────────────────────────────────────────────────
if st.button("▶ Run Forecast", type="primary"):
    with st.spinner("Running recursive forecast..."):
        forecast_df = recursive_forecast(
            model, df_train["unit_sales"], n_days,
            feature_cols=FEATURE_COLS,
            build_features_fn=build_features,
            get_holiday_features_fn=get_holiday_features,
            holiday_features=holiday_features
        )

    actuals = df_test["unit_sales"].reindex(forecast_df.index)
    overlap_pred = forecast_df["forecast"].reindex(actuals.dropna().index)
    overlap_true = actuals.dropna()

    # ── Metrics (validation mode only) ───────────────────────────────────────
    if show_actuals and len(overlap_true) > 0:
        mae   = np.mean(np.abs(overlap_pred - overlap_true))
        rmse  = np.sqrt(np.mean((overlap_pred - overlap_true) ** 2))
        smape = np.mean(
            2 * np.abs(overlap_pred - overlap_true) /
            (np.abs(overlap_pred) + np.abs(overlap_true) + 1e-8)
        ) * 100

        st.markdown(f"""
            <div style='display:flex; gap:16px; margin:16px 0;'>
                <div style='flex:1; border:1px solid #ddd; border-radius:6px; padding:16px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>MAE</div>
                    <div style='font-size:24px; font-weight:600; color:#3d3d3d;'>{mae:.2f}</div>
                    <div style='font-size:12px; color:#aaa;'>units</div>
                </div>
                <div style='flex:1; border:1px solid #ddd; border-radius:6px; padding:16px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>RMSE</div>
                    <div style='font-size:24px; font-weight:600; color:#3d3d3d;'>{rmse:.2f}</div>
                    <div style='font-size:12px; color:#aaa;'>units</div>
                </div>
                <div style='flex:1; border:1px solid #ddd; border-radius:6px; padding:16px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>sMAPE</div>
                    <div style='font-size:24px; font-weight:600; color:#3d3d3d;'>{smape:.2f}%</div>
                    <div style='font-size:12px; color:#aaa;'>symmetric</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Plot forecast ─────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    history_window = df_train["unit_sales"].last("60D")
    ax2.plot(history_window.index, history_window.values,
            label="Training history", color="steelblue", linewidth=1.5)
    if show_actuals:
        ax2.plot(actuals.index, actuals.values,
                label="Actual sales (test)", color="seagreen", linewidth=1.5)
    ax2.plot(forecast_df.index, forecast_df["forecast"].values,
            label=f"XGBoost forecast ({n_days} days)", color="tomato",
            linewidth=2, linestyle="--", marker="o", markersize=3)
    ax2.axvline(pd.Timestamp("2014-01-01"), color="gray",
                linestyle=":", alpha=0.7, label="Forecast start")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Unit Sales")
    ax2.set_title(
        f"XGBoost Recursive Forecast vs Actual Sales — {n_days} Days"
        if show_actuals else
        f"XGBoost Recursive Forecast — {n_days} Days"
    )
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Best & Worst day + Error plot (validation mode only) ──────────────────
    if show_actuals and len(overlap_true) > 0:
        daily_error = np.abs(overlap_pred - overlap_true)
        best_day    = daily_error.idxmin()
        worst_day   = daily_error.idxmax()

        st.markdown(f"""
            <div style='display:flex; gap:16px; margin:16px 0;'>
                <div style='flex:1; border:1px solid #ddd; border-radius:6px; padding:16px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>✅ Best Forecast Day</div>
                    <div style='font-size:22px; font-weight:600; color:#3d3d3d;'>{best_day.strftime("%Y-%m-%d")}</div>
                    <div style='font-size:12px; color:#aaa;'>Error: {daily_error[best_day]:.1f} units</div>
                </div>
                <div style='flex:1; border:1px solid #ddd; border-radius:6px; padding:16px; text-align:center;'>
                    <div style='font-size:13px; color:#888;'>⚠️ Worst Forecast Day</div>
                    <div style='font-size:22px; font-weight:600; color:#3d3d3d;'>{worst_day.strftime("%Y-%m-%d")}</div>
                    <div style='font-size:12px; color:#aaa;'>Error: {daily_error[worst_day]:.1f} units</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Daily error plot
        fig3, ax3 = plt.subplots(figsize=(12, 3))
        ax3.bar(daily_error.index, daily_error.values,
                color="tomato", alpha=0.7, width=0.8)
        ax3.axhline(mae, color="gray", linestyle="--", linewidth=1,
                    label=f"Mean MAE = {mae:.1f}")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Absolute Error (units)")
        ax3.set_title("Daily Forecast Error")
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown(
            """<p style='color:#555; font-size:14px;'>
            ⚠️ Forecast errors are higher around atypical high-demand days not
            associated with known holidays or regular weekly patterns. Error
            stabilizes once demand returns to its regular weekly structure.
            </p>""",
            unsafe_allow_html=True
        )

    # ── Forecast table ────────────────────────────────────────────────────────
    st.markdown("#### Forecast Table")
    comparison = forecast_df.copy()
    comparison = comparison.rename(columns={"forecast": "Forecast"})
    if show_actuals:
        comparison["Actual Sales"] = actuals
    comparison.index = comparison.index.strftime("%Y-%m-%d")
    st.dataframe(comparison, use_container_width=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    st.download_button(
        label="⬇ Download Forecast CSV",
        data=comparison.to_csv(),
        file_name=f"favorita_forecast_{n_days}d.csv",
        mime="text/csv"
    )