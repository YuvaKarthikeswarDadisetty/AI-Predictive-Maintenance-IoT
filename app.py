# ============================================
# STREAMLIT UI - CLEAN + PROFESSIONAL VERSION
# NASA Predictive Maintenance
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ============================================
# LOAD & PREPROCESS NASA DATA
# ============================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)

    columns = ["engine_id", "cycle"] + [f"sensor_{i}" for i in range(1, df.shape[1]-1)]
    df.columns = columns

    return df


def add_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    return df


def create_failure_label(df, threshold=30):
    df["failure"] = np.where(df["RUL"] <= threshold, 1, 0)
    return df


def prepare_data(df):
    df = df.drop(["engine_id", "cycle", "max_cycle", "RUL"], axis=1)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, y, scaler, X_scaled


# ============================================
# LOAD DATA
# ============================================

DATA_PATH = "data/nasa/train_FD001.txt"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset not found! Place it in data/nasa/")
    st.stop()

df = load_data(DATA_PATH)
df = add_rul(df)
df = create_failure_label(df)

X, y, scaler, X_scaled = prepare_data(df)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)


# ============================================
# UI DESIGN
# ============================================

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

st.title("🔧 AI Predictive Maintenance Dashboard")
st.markdown("### 🚀 NASA Turbofan Engine Dataset")

st.markdown("Adjust sensor values and predict machine failure.")

# ============================================
# CLEAN SENSOR SELECTION (IMPORTANT)
# ============================================

selected_features = [
    "sensor_2",  # temperature
    "sensor_3",  # vibration
    "sensor_4",  # pressure
    "sensor_7",  # humidity
    "sensor_11"  # additional important sensor
]

st.subheader("🎛️ Input Sensor Values")

input_data = []

for col in selected_features:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())

    # FIX for constant values
    if min_val == max_val:
        st.warning(f"{col} has constant value. Using default.")
        val = min_val
    else:
        val = st.slider(col, min_val, max_val, mean_val)

    input_data.append(val)

# Fill remaining sensors with mean values
for col in X.columns:
    if col not in selected_features:
        input_data.append(float(X[col].mean()))

# Ensure correct order
input_df = pd.DataFrame([input_data], columns=X.columns)


# ============================================
# PREDICTION
# ============================================

if st.button("🔍 Predict Machine Status"):

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("✅ Machine is Healthy")
    else:
        st.error("🚨 Machine Failure Likely!")


# ============================================
# DATA VISUALIZATION
# ============================================

st.subheader("📊 Failure Distribution")

st.bar_chart(df["failure"].value_counts())


# ============================================
# DATA PREVIEW
# ============================================

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("💡 Built using Machine Learning + NASA CMAPSS Dataset")