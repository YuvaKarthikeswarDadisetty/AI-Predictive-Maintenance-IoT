# ============================================
# AI-Powered Predictive Maintenance (FINAL)
# NASA Dataset + ML + Visualizations + Prediction
# ============================================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# STEP 1: LOAD NASA DATA
# ============================================

def load_nasa_data(path):
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


def prepare_features(df):
    df = df.drop(["engine_id", "cycle", "max_cycle", "RUL"], axis=1)

    X = df.drop("failure", axis=1)
    y = df["failure"]

    return X, y


# ============================================
# LOAD DATASET
# ============================================

file_path = "data/nasa/train_FD001.txt"

if not os.path.exists(file_path):
    print("❌ ERROR: Dataset not found!")
    print("Place it here: data/nasa/train_FD001.txt")
    exit()

df = load_nasa_data(file_path)

print("✅ NASA Dataset Loaded")
print(df.head())


# ============================================
# PREPROCESSING
# ============================================

df = add_rul(df)
df = create_failure_label(df)

X, y = prepare_features(df)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n✅ Preprocessing Completed")
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ============================================
# MODEL TRAINING
# ============================================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n🤖 Model Training Completed")


# ============================================
# PREDICTION
# ============================================

y_pred = model.predict(X_test)

print("\n🔍 Sample Predictions:", y_pred[:10])


# ============================================
# EVALUATION
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n📈 Accuracy:", accuracy)

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))


# ============================================
# CREATE IMAGES FOLDER
# ============================================

os.makedirs("images", exist_ok=True)


# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("images/confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved")


# ============================================
# VISUALIZATION (NASA → MAPPED FEATURES)
# ============================================

df_visual = df.copy()

# Map sensors
df_visual["temperature"] = df_visual["sensor_2"]
df_visual["vibration"] = df_visual["sensor_3"]
df_visual["pressure"] = df_visual["sensor_4"]
df_visual["humidity"] = df_visual["sensor_7"]
df_visual["runtime_hours"] = df_visual["cycle"]

features = ["temperature", "vibration", "pressure", "humidity", "runtime_hours"]

# --------------------------------------------
# SENSOR DISTRIBUTIONS
# --------------------------------------------

for feature in features:
    plt.figure()
    sns.histplot(df_visual[feature], kde=True)
    plt.title(f"{feature} Distribution")
    plt.savefig(f"images/{feature}_distribution.png")
    plt.close()

print("✅ Sensor distribution plots saved")


# --------------------------------------------
# FAILURE DISTRIBUTION
# --------------------------------------------

plt.figure()
sns.countplot(x="failure", data=df_visual)
plt.title("Failure vs Non-Failure")
plt.savefig("images/failure_distribution.png")
plt.close()

print("✅ Failure distribution saved")


# --------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------

plt.figure(figsize=(10, 8))
sns.heatmap(df_visual.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.close()

print("✅ Correlation heatmap saved")


# ============================================
# REAL-TIME PREDICTION
# ============================================

def predict_machine_status(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        print("\n✅ Machine is Healthy")
    else:
        print("\n🚨 ALERT: Machine Failure Likely!")

    return prediction


print("\n🔧 Testing Real-Time Prediction...")

sample_input = X.iloc[0].values.tolist()
predict_machine_status(sample_input)


# ============================================
# FINAL MESSAGE
# ============================================

print("\n🎉 FULL PROJECT COMPLETED WITH NASA DATA 🚀")