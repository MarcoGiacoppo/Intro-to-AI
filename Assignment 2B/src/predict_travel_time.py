import numpy as np
from datetime import datetime
from keras.models import load_model
import joblib

# === Load model and scalers ===
MODEL_PATH = "../models/lstm_model.h5"
SCALER_PATH = "../models/scaler.pkl"

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

def predict_travel_time(scats_id: str, hour: int = None) -> float:
    """
    Predicts traffic volume using trained LSTM and converts to estimated travel time in minutes per km.
    """
    if hour is None:
        hour = datetime.now().hour

    try:
        scats_id = int(scats_id)
    except:
        return 1.0  # fallback in case of bad input

    # Prepare input
    input_array = np.array([[scats_id, hour]])
    input_scaled = scaler.transform(input_array)
    input_seq = np.expand_dims(input_scaled, axis=0)  # shape: (1, 1, 2)

    # Predict
    predicted_volume = model.predict(input_seq, verbose=0)[0][0]

    # Convert volume to speed → travel time
    a, b = 0.001, 1.6  # from the parabolic formula PDF
    speed = max(5.0, 100 * (1 - a * (predicted_volume ** b)))
    travel_time = round(60 / speed, 2)

    return travel_time
