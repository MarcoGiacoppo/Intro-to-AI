import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# === CONFIG ===
CSV_PATH = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"
MODEL_PATH = "../models/tcn_model.h5"
SCATS_ID = "0970"
SEQ_LENGTH = 24
PRED_OFFSET = 1

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))
site_df = df[df["SCATS Number"] == SCATS_ID].sort_values("Date")

# Get V00–V95 columns and flatten into time series
v_cols = [col for col in site_df.columns if col.startswith("V")]
values = site_df[v_cols].values.flatten()

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values.reshape(-1, 1))

# Take the last 24 hours
X_input = scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

# === PREDICT ===
from tcn import TCN
model = load_model(MODEL_PATH, compile=False, custom_objects={"TCN": TCN})
pred_scaled = model.predict(X_input)
pred_volume = scaler.inverse_transform(pred_scaled)[0][0]

# === CONVERT TO TRAVEL TIME ===
# Parabolic formula: speed = a * volume² + b * volume + c
# Adjust these coefficients based on the Traffic Flow to Travel Time PDF if needed
a, b, c = -0.0006, 0.24, 60
speed = a * (pred_volume ** 2) + b * pred_volume + c
speed = max(speed, 1)  # Clamp to avoid division by zero or negative speed
travel_time_min_per_km = 60 / speed

# === DISPLAY RESULT ===
print(f"🔍 Predicted traffic volume: {pred_volume:.2f}")
print(f"🚗 Estimated speed: {speed:.2f} km/h")
print(f"⏱ Estimated travel time: {travel_time_min_per_km:.2f} minutes per km")
