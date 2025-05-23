import numpy as np
import json
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import pandas as pd
import math
from keras.saving import register_keras_serializable # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tcn import TCN

@register_keras_serializable(package="Custom")
class RegisteredTCN(TCN):
    pass

@register_keras_serializable(package="Custom")
def squeeze_dim1(x):
    return K.squeeze(x, axis=1)

def load_metadata_and_adjacency():
    with open("../data/graph/sites_metadata.json") as f:
        metadata = json.load(f)
    with open("../data/graph/adjacency_from_summary.json") as f:
        adjacency = json.load(f)
    return metadata, adjacency

# Load when script is run (or explicitly from outside)
try:
    metadata, adjacency = load_metadata_and_adjacency()
except FileNotFoundError:
    metadata, adjacency = {}, {}

site_ids = sorted(metadata.keys())
travel_time_cache = {}
_model_cache = {}

# === Load traffic flow CSV once (global!) ===
traffic_df = pd.read_csv("../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv")
traffic_df["SCATS Number"] = traffic_df["SCATS Number"].apply(lambda x: str(x).zfill(4))
v_cols = [col for col in traffic_df.columns if col.startswith("V") and col[1:].isdigit()]
traffic_df_long = traffic_df.melt(id_vars=["SCATS Number", "Date"], value_vars=v_cols,
                                   var_name="Bin", value_name="volume")
traffic_df_long["hour"] = traffic_df_long["Bin"].apply(lambda x: int(x[1:]) // 4)
traffic_df_long["minute_offset"] = traffic_df_long["Bin"].apply(lambda x: int(x[1:]) * 15)
traffic_df_long["timestamp"] = pd.to_datetime(traffic_df_long["Date"]) + pd.to_timedelta(traffic_df_long["minute_offset"], unit="m")

# === Load model and encoders ===
def load_prediction_components(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name == "tcn":
        model = load_model(
            f"../models/{model_name}_model.keras", 
            compile=False,
            custom_objects={"RegisteredTCN": RegisteredTCN}
        )
    else:
        model = load_model(f"../models/{model_name}_model.keras", compile=False)

    scaler = joblib.load(f"../models/{model_name}_scaler.pkl")
    encoder = joblib.load(f"../models/{model_name}_scats_encoder.pkl")

    _model_cache[model_name] = (model, scaler, encoder)
    return model, scaler, encoder


def flow_to_travel_time(volume, distance_km=1.0):
    try:
        if volume is None or volume < 0:
            return 12.0

        a, b, c = -1.4648375, 93.75, -volume
        d = b**2 - 4 * a * c

        if d < 0:
            return 12.0

        speed = (-b - math.sqrt(d)) / (2 * a)

        # Clamp only absurd values, not reasonable speeds
        if speed > 60:
            #Instead of clamping to 60:
            speed *= 0.9
        speed = max(5.0, speed)

        return round((distance_km / speed) * 60 + 0.5, 2)

    except Exception as e:
        print(f"[FLOW2TIME ERROR] volume={volume} → {e}")
        return 12.0

# === Efficient real input sequence ===
def get_real_input_sequence(scats_id, scaler, selected_hour=None):
    """
    Returns a valid 24-step input sequence for the selected SCATS site and hour.
    Adds debug output to verify variability.
    """
    try:
        df = traffic_df_long[traffic_df_long["SCATS Number"] == scats_id]

        if df.empty:
            print(f"[WARN] No data for SCATS {scats_id}")
            return None

        if selected_hour is None:
            selected_hour = datetime.now().hour

        # Filter strictly by selected hour
        df_hour = df[df["hour"] == selected_hour].sort_values("timestamp")

        if df_hour.empty:
            print(f"[FALLBACK] No entries for SCATS {scats_id} at hour {selected_hour}")
            return None

        # Get the latest timestamp within this hour
        latest_ts = df_hour["timestamp"].max()

        # Get the 24 values before that point
        seq_df = df[df["timestamp"] < latest_ts].sort_values("timestamp").tail(24)

        if len(seq_df) < 24:
            print(f"[FALLBACK] Not enough data before {latest_ts} for SCATS {scats_id}")
            return None

        volumes = seq_df["volume"].values
        scaled = scaler.transform(volumes.reshape(-1, 1))

        return scaled.reshape(1, 24, 1)

    except Exception as e:
        print(f"[ERROR] get_real_input_sequence failed for SCATS {scats_id} @ hour {selected_hour}: {e}")
        return None

# === Predict travel time (called only in preload step now) ===
def predict_travel_time_model(scats_id, model_name="lstm", hour_override=None):
    scats_id = str(int(scats_id))
    model, scaler, encoder = load_prediction_components(model_name)

    if scats_id not in encoder.classes_:
        return 12.0  # fallback if encoder can't map the site

    idx = encoder.transform([scats_id])[0]
    input_scats = np.array([[idx]])
    input_seq = get_real_input_sequence(scats_id, scaler, hour_override)
    if input_seq is None:
        return 12.0

    pred_scaled = model.predict([input_seq, input_scats], verbose=0)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]
    pred_volume = max(0.0, min(pred_volume, 400))  # Clamp [0, 400]

    # Assume average segment distance = 1.0 km
    return flow_to_travel_time(pred_volume, 1.0)

# === Preload all travel times once per model/hour ===
def preload_all_travel_times(model_name, selected_hour):
    model, scaler, encoder = load_prediction_components(model_name)
    preload_cache = {}

    for scats_id in traffic_df["SCATS Number"].unique():
        if scats_id not in encoder.classes_:
            continue

        idx = encoder.transform([scats_id])[0]
        input_scats = np.array([[idx]])
        input_seq = get_real_input_sequence(scats_id, scaler, selected_hour)
        if input_seq is None:
            continue

        pred_scaled = model.predict([input_seq, input_scats], verbose=0)[0][0]
        pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]
        pred_volume = max(0.0, min(pred_volume, 400))  # Clamp [0, 400]

        travel_time = flow_to_travel_time(pred_volume, 1.0)

        preload_cache[(scats_id, model_name, selected_hour)] = travel_time

    return preload_cache

# === Haversine util ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Neighbors & heuristics ===
def get_neighbors(node):
    return adjacency.get(str(int(node)), [])

def heuristic_fn(n, goal):
    n, goal = str(int(n)), str(int(goal))
    if n not in metadata or goal not in metadata:
        return 0
    m1, m2 = metadata[n], metadata[goal]
    if not all([m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"]]):
        return 0
    return haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])

# === Final cost_fn: ultra fast
def cost_fn(a, b, model_name, selected_hour):
    a, b = str(int(a)), str(int(b))
    if a not in metadata or b not in metadata:
        return float("inf")

    key = (b, model_name, selected_hour)
    travel_time = travel_time_cache.get(key)
    if travel_time is None:
        return float("inf")

    m1, m2 = metadata[a], metadata[b]
    if not all([m1.get("latitude"), m1.get("longitude"), m2.get("latitude"), m2.get("longitude")]):
        return float("inf")

    dist = haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])

    return round(travel_time * dist, 2)  # Final units: minutes

# === Total distance calculator ===
def calculate_total_distance(path):
    total_km = 0.0
    for i in range(1, len(path)):
        a, b = str(path[i - 1]), str(path[i])
        if a not in metadata or b not in metadata:
            continue
        lat1, lon1 = metadata[a]["latitude"], metadata[a]["longitude"]
        lat2, lon2 = metadata[b]["latitude"], metadata[b]["longitude"]
        total_km += haversine(lat1, lon1, lat2, lon2)
    return round(total_km, 2)