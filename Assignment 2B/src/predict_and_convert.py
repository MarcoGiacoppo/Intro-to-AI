import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import joblib
import sys
import os

# === Get SCATS ID from CLI or input ===
if len(sys.argv) > 1:
    SCATS_ID = sys.argv[1].zfill(4)
else:
    SCATS_ID = input("Enter SCATS ID (e.g., 0970): ").strip().zfill(4)

# === CONFIG ===
SEQ_LENGTH = 24
MODEL_NAME = "tcn"
MODEL_PATH = f"../models/{MODEL_NAME}_model.keras"
SCALER_PATH = f"../models/{MODEL_NAME}_scaler.pkl"
ENCODER_PATH = f"../models/{MODEL_NAME}_scats_encoder.pkl"
CSV_PATH = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"

# === LOAD MODEL, SCALER, ENCODER ===
from tcn import TCN
model = load_model(MODEL_PATH, compile=False, custom_objects={"TCN": TCN})
scaler = joblib.load(SCALER_PATH)
scats_encoder = joblib.load(ENCODER_PATH)

# === Load and prepare data ===
df = pd.read_csv(CSV_PATH)
df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))

if SCATS_ID not in df["SCATS Number"].values:
    print(f"❌ SCATS site {SCATS_ID} not found in dataset.")
    exit()

site_df = df[df["SCATS Number"] == SCATS_ID].sort_values("Date")
v_cols = [col for col in site_df.columns if col.startswith("V")]
values = site_df[v_cols].values.flatten()

scaled = scaler.transform(values.reshape(-1, 1))
X_input = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

# === Encode SCATS site input ===
scats_idx = scats_encoder.transform([SCATS_ID])[0]
X_scats = np.array([[scats_idx]])

# === Predict volume and convert ===
pred_scaled = model.predict([X_input, X_scats], verbose=0)
pred_volume = scaler.inverse_transform(pred_scaled)[0][0]

# === Volume to speed and travel time ===
# (Parabolic formula: speed = a * v² + b * v + c)
a, b, c = -0.0006, 0.24, 60
speed = a * (pred_volume ** 2) + b * pred_volume + c
speed = max(speed, 1.0)
travel_time = 60 / speed

# === Output ===
print(f"📍 SCATS site: {SCATS_ID}")
print(f"🔍 Predicted traffic volume: {pred_volume:.2f}")
print(f"🚗 Estimated speed: {speed:.2f} km/h")
print(f"⏱ Estimated travel time: {travel_time:.2f} minutes per km")
