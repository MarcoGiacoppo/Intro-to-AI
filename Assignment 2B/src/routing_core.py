import numpy as np
import json
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from tensorflow.keras.models import load_model  # type: ignore
import joblib

# === Load data once ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)
with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

site_ids = sorted(metadata.keys())
travel_time_cache = {}
_model_cache = {}

# === Model prediction ===
def load_prediction_components(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    model = load_model(f"../models/{model_name}_model.keras", compile=False)
    scaler = joblib.load(f"../models/{model_name}_scaler.pkl")
    encoder = joblib.load(f"../models/{model_name}_scats_encoder.pkl")

    _model_cache[model_name] = (model, scaler, encoder)
    return model, scaler, encoder

def predict_travel_time_model(scats_id, model_name="lstm"):
    scats_id = str(int(scats_id))
    hour = datetime.now().hour

    model, scaler, encoder = load_prediction_components(model_name)

    if scats_id not in encoder.classes_:
        return 1.0

    idx = encoder.transform([scats_id])[0]
    dummy_seq = np.full((1, 24, 1), 0.5)
    input_scats = np.array([[idx]])

    pred_scaled = model.predict([dummy_seq, input_scats], verbose=0)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]

    a, b = 0.001, 1.6
    speed = max(5.0, 100 * (1 - a * (pred_volume ** b)))
    return round(60 / speed, 2)


# === Routing utils ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

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

def cost_fn(a, b, model_name):
    a, b = str(int(a)), str(int(b))

    if a not in metadata or b not in metadata:
        return float("inf")

    if b in travel_time_cache:
        travel_time = travel_time_cache[b]
    else:
        travel_time = predict_travel_time_model(b, model_name)
        travel_time_cache[b] = travel_time

    m1, m2 = metadata[a], metadata[b]
    if not all([m1.get("latitude"), m1.get("longitude"), m2.get("latitude"), m2.get("longitude")]):
        return float("inf")

    if travel_time > 240:
        return float("inf")

    dist = haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])
    return dist * travel_time

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
