import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.predict import predict_machine
from src.visualize import save_confusion_matrix

# Generate dataset (optional)
np.random.seed(42)
n = 120

df = pd.DataFrame({
    "temperature": np.random.randint(40, 95, n),
    "vibration": np.round(np.random.uniform(0.01, 0.12, n), 3),
    "pressure": np.random.randint(25, 75, n),
    "humidity": np.random.randint(30, 65, n),
    "runtime_hours": np.random.randint(50, 400, n)
})

df["failure"] = np.where(
    (df["temperature"] > 75) &
    (df["vibration"] > 0.08) &
    (df["pressure"] > 55),
    1, 0
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/sensor_data.csv", index=False)

# Load
df = load_data()

# Preprocess
X, y, scaler = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = train_model(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc, report, cm = evaluate_model(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nReport:\n", report)

# Save confusion matrix
save_confusion_matrix(cm)

# Real-time prediction
columns = df.drop("failure", axis=1).columns

print("\n🔧 Real-time Test:")
prediction = predict_machine(model, scaler, [85, 0.1, 65, 55, 300], columns)

if prediction == 1:
    print("🚨 Failure Likely")
else:
    print("✅ Healthy")