import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib
import sys
import os

# === CONFIG ===
SEQ_LENGTH = 24
MODEL_NAME = "lstm"  # Change to 'gru' or 'tcn' if needed
MODEL_PATH = f"../models/{MODEL_NAME}_model.keras"
SCALER_PATH = f"../models/{MODEL_NAME}_scaler.pkl"
ENCODER_PATH = f"../models/{MODEL_NAME}_scats_encoder.pkl"

# === Load model, scaler, encoder ===
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
scats_encoder = joblib.load(ENCODER_PATH)

def predict_travel_time(scats_id: str, hour: int = None) -> float:
    """
    Predicts travel time per km for a SCATS site using the selected ML model.
    Note: `hour` is accepted for compatibility but not used in the model.
    """
    if hour is None:
        hour = datetime.now().hour  # Note: hour is unused in current model

    scats_id = str(scats_id).zfill(4)
    if scats_id not in scats_encoder.classes_:
        return 1.0  # Fallback if unknown SCATS site

    scats_idx = scats_encoder.transform([scats_id])[0]

    # Generate dummy sequence — always the same input
    dummy_seq = np.full((1, SEQ_LENGTH, 1), 0.5)
    input_scats = np.array([[scats_idx]])

    # Predict scaled volume
    pred_scaled = model.predict([dummy_seq, input_scats], verbose=0)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]

    # Convert volume → speed → travel time (min/km)
    a, b = 0.001, 1.6
    speed = max(5.0, 100 * (1 - a * (pred_volume ** b)))
    travel_time = round(60 / speed, 2)

    return travel_time

# === CLI test ===
if __name__ == "__main__":
    scats_id = input("Enter SCATS ID (e.g., 0970): ").strip().zfill(4)
    try:
        hour = int(input("Enter hour (0–23): ").strip())
    except ValueError:
        hour = datetime.now().hour

    print(f"⏱ Travel time estimate: {predict_travel_time(scats_id, hour):.2f} min/km")
