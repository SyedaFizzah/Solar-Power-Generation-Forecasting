# src/streamlit_app_updated.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy import stats
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from pathlib import Path

# Optional: TCN import
try:
    from tcn import TCN
except ImportError:
    TCN = None

# ----------------- Streamlit page setup -----------------
st.set_page_config(page_title="Solar Power Drift Monitor", layout="wide")
st.title("Solar Power Forecasting + Real-Time Drift Detection")
st.markdown("**Upload hourly CSV**")

# ----------------- Paths -----------------
BASE_DIR = Path("D:/fizzah/deep learning/improved_models")
X_ref_path = BASE_DIR / "X_ref.npy"
y_ref_path = BASE_DIR / "y_ref.npy"
scaler_X_path = BASE_DIR / "scaler_X.save"
scaler_y_path = BASE_DIR / "scaler_y.save"
xgb_path = BASE_DIR / "Teacher_XGBoost.joblib"
cnn_path = BASE_DIR / "1D_CNN_model.keras"
lstm_path = BASE_DIR / "LSTM_model.keras"
tcn_path = BASE_DIR / "TCN_model.keras"

# ----------------- Load references -----------------
@st.cache_resource
def load_refs():
    X_ref = np.load(X_ref_path)
    y_ref = np.load(y_ref_path)
    return X_ref, y_ref

# ----------------- Load models & scalers -----------------
@st.cache_resource
def load_models():
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    xgb_teacher = joblib.load(xgb_path)

    cnn = tf.keras.models.load_model(cnn_path, compile=False)
    lstm = tf.keras.models.load_model(lstm_path, compile=False)

    tcn_model = None
    if TCN and tcn_path.exists():
        tcn_model = tf.keras.models.load_model(tcn_path, custom_objects={'TCN': TCN}, compile=False)

    return scaler_X, scaler_y, xgb_teacher, cnn, lstm, tcn_model

# ----------------- Load references and models -----------------
X_ref, y_ref = load_refs()
scaler_X, scaler_y, xgb_teacher, cnn, lstm, tcn = load_models()

# ----------------- Configuration -----------------
LOOKBACK = st.sidebar.number_input("LOOKBACK window (hours)", min_value=24, max_value=720, value=168, step=24)

# ----------------- Upload CSV -----------------
uploaded_file = st.file_uploader("Upload your hourly data (CSV only)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV to activate the dashboard")
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Validate timestamp
    if "utc_timestamp" not in df.columns:
        st.error("Missing 'utc_timestamp' column!")
        st.stop()

    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    df = df.set_index("utc_timestamp").tz_convert(None).sort_index()

    # Cyclical features
    for col, period in [("hour",24), ("dayofweek",7), ("month",12)]:
        df[col] = getattr(df.index, col.split("_")[0] if "_" in col else col)
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # Prepare data
    TARGET = "FR_solar_generation_actual"
    feature_cols = getattr(scaler_X, "feature_names_in_", None) or [c for c in df.columns if c != TARGET]
    X_raw = df[feature_cols].values
    X_scaled = scaler_X.transform(X_raw)
    y_raw = np.clip(df[TARGET].values, 0, None)
    y_scaled = scaler_y.transform(np.log1p(y_raw).reshape(-1,1))

    # Sequences
    X_seq = np.array([X_scaled[i:i+LOOKBACK] for i in range(len(X_scaled)-LOOKBACK)])
    y_seq = y_scaled[LOOKBACK:]
    actual_mw = np.expm1(scaler_y.inverse_transform(y_seq)).flatten()

    # Prediction function
    def pred(model, X):
        if model is None: return np.zeros(len(X))
        return np.clip(np.expm1(scaler_y.inverse_transform(model.predict(X, verbose=0))), 0, None).flatten()

    cnn_p = pred(cnn, X_seq)
    lstm_p = pred(lstm, X_seq)
    tcn_p = pred(tcn, X_seq) if tcn else np.zeros_like(cnn_p)

    # Teacher model features
    base = np.column_stack([cnn_p, lstm_p, tcn_p])
    teacher_X = np.hstack([
        X_raw[LOOKBACK:], base,
        base.mean(1, keepdims=True), base.std(1, keepdims=True),
        base.min(1, keepdims=True), base.max(1, keepdims=True)
    ])
    forecast_mw = xgb_teacher.predict(teacher_X).clip(0)

    # Drift detection
    def ks_target_drift(y_ref, y_new, alpha=0.01):
        _, p = stats.ks_2samp(y_ref, y_new)
        return p < alpha, p

    def mmd_feature_drift(X_ref, X_new, gamma=0.5, subsample=1000, n_perm=200, alpha=0.01):
        X_ref_sub = X_ref if len(X_ref) <= subsample else X_ref[np.random.choice(len(X_ref), subsample, replace=False)]
        X_new_sub = X_new if len(X_new) <= subsample else X_new[np.random.choice(len(X_new), subsample, replace=False)]
        Kxx = pairwise_kernels(X_ref_sub, X_ref_sub, metric='rbf', gamma=gamma)
        Kyy = pairwise_kernels(X_new_sub, X_new_sub, metric='rbf', gamma=gamma)
        Kxy = pairwise_kernels(X_ref_sub, X_new_sub, metric='rbf', gamma=gamma)
        mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

        combined = np.vstack([X_ref_sub, X_new_sub])
        n = len(X_ref_sub)
        mmd_perms = []
        for _ in range(n_perm):
            idx = np.random.permutation(2*n)
            A, B = combined[idx[:n]], combined[idx[n:]]
            mmd_perms.append(pairwise_kernels(A,A,metric='rbf',gamma=gamma).mean() +
                             pairwise_kernels(B,B,metric='rbf',gamma=gamma).mean() -
                             2*pairwise_kernels(A,B,metric='rbf',gamma=gamma).mean())
        p_value = (np.array(mmd_perms) >= mmd2).mean() + 1/(n_perm+1)
        return p_value < alpha, mmd2, p_value

    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    y_flat = y_seq.flatten()

    target_drift, p_target = ks_target_drift(y_ref, y_flat)
    feature_drift, mmd2, p_feat = mmd_feature_drift(X_ref, X_flat)

    # Display metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Gen", f"{actual_mw.sum()/1e6:.2f} TWh")
    c2.metric("Avg Hourly", f"{actual_mw.mean():.0f} MW")
    c3.metric("Feature Drift", "YES" if feature_drift else "NO", delta=f"MMD²={mmd2:.4f}, p={p_feat:.4f}")
    c4.metric("Target Drift", "YES" if target_drift else "NO", delta=f"p={p_target:.6f}")

    if feature_drift or target_drift:
        st.warning("Drift detected! Model may underperform. Retraining recommended.")

    # ---------------- Forecast Accuracy ----------------
    mse = mean_squared_error(actual_mw, forecast_mw)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_mw, forecast_mw)
    st.write(f"Forecast RMSE: {rmse:.2f} MW, MAE: {mae:.2f} MW")

    # ---------------- Plot last N hours ----------------
    last_n = min(500, len(df_plot := pd.DataFrame({"Actual": actual_mw, "Forecast": forecast_mw}, index=df.index[LOOKBACK:])))
    st.plotly_chart(
        go.Figure()
          .add_trace(go.Scatter(x=df_plot.index[-last_n:], y=df_plot.Actual[-last_n:], name="Actual"))
          .add_trace(go.Scatter(x=df_plot.index[-last_n:], y=df_plot.Forecast[-last_n:], name="Forecast", line=dict(dash="dot")))
          .update_layout(title=f"Last {last_n} hours"),
        use_container_width=True
    )

    # ---------------- Drift visuals ----------------
    st.subheader("Drift Visualization")
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=y_ref.flatten(), nbinsx=50, name='Reference', opacity=0.7))
    fig1.add_trace(go.Histogram(x=y_flat, nbinsx=50, name='New', opacity=0.7))
    fig1.update_layout(title="Target Distribution Drift", barmode='overlay')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=X_ref[:,0], nbinsx=50, name='Reference', opacity=0.7))
    fig2.add_trace(go.Histogram(x=X_flat[:,0], nbinsx=50, name='New', opacity=0.7))
    fig2.update_layout(title=f"Feature Distribution Drift: {feature_cols[0]}", barmode='overlay')
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Monthly aggregated ----------------
    monthly = df_plot.resample("MS").sum() / 1e6
    st.plotly_chart(
        go.Figure()
          .add_trace(go.Bar(x=monthly.index, y=monthly.Actual, name="Actual"))
          .add_trace(go.Bar(x=monthly.index, y=monthly.Forecast, name="Forecast"))
          .update_layout(barmode="group", title="Monthly Generation (TWh)"),
        use_container_width=True
    )
