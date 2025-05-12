import streamlit as st
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model #type:ignore
import joblib
from search_algorithms import dfs, bfs, ucs, astar
from display_route_map import display_route_map
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

# === Load metadata and adjacency ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)
with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

site_ids = sorted(metadata.keys())

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
    n = str(int(n))
    goal = str(int(goal))
    if n not in metadata or goal not in metadata:
        return 0
    m1, m2 = metadata[n], metadata[goal]
    if not all([m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"]]):
        return 0
    return haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])

@st.cache_resource(show_spinner=False)
def load_prediction_components(model_name):
    model = load_model(f"../models/{model_name}_model.keras", compile=False)
    scaler = joblib.load(f"../models/{model_name}_scaler.pkl")
    encoder = joblib.load(f"../models/{model_name}_scats_encoder.pkl")
    return model, scaler, encoder

def predict_travel_time_model(scats_id, model_name="lstm"):
    scats_id = str(int(scats_id))  # Normalize by removing leading zeros
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

travel_time_cache = {}

def cost_fn(a, b):
    a = str(int(a))
    b = str(int(b))

    if a not in metadata or b not in metadata:
        return float("inf")

    if b in travel_time_cache:
        travel_time = travel_time_cache[b]
    else:
        travel_time = predict_travel_time_model(b, st.session_state.model_choice)
        travel_time_cache[b] = travel_time

    m1, m2 = metadata[a], metadata[b]
    if not all([m1.get("latitude"), m1.get("longitude"), m2.get("latitude"), m2.get("longitude")]):
        return float("inf")

    if travel_time > 240:
        return float("inf")

    dist = haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])
    return dist * travel_time

# === Streamlit UI ===
st.set_page_config(page_title="TBRGS - Route Finder", layout="wide")

st.markdown("""
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 1rem;
        padding-bottom: 1rem;
        margin: auto;
    }
    .stSelectbox > div, .stDateInput > div, .stNumberInput > div {
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }
    .stButton button {
        padding: 0.4rem 0.9rem;
        font-size: 0.88rem;
    }
    .stMarkdown h2, .stMarkdown h3 {
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    .stDataFrameContainer {
        padding: 0rem;
    }
            
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>🛣️ Traffic-Based Route Guidance System</h1>", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = {}

st.markdown("## 🚦 Select Route Parameters")

# Centered layout
left_pad, center, right_pad = st.columns([0.1, 2.2, 0.1])

with center:
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown('<div style="width:100%">', unsafe_allow_html=True)
            origin = st.selectbox("🛫 Origin SCATS ID", site_ids, index=site_ids.index("970"), key="origin")
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        with st.container():
            st.markdown('<div style="width:100%">', unsafe_allow_html=True)
            destination = st.selectbox("🏁 Destination SCATS ID", site_ids, index=site_ids.index("2000"), key="destination")
            st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        with st.container():
            st.markdown('<div style="width:100%">', unsafe_allow_html=True)
            model_choice = st.selectbox("🧠 Prediction Model", ["lstm", "gru", "tcn"], key="model")
            st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        with st.container():
            st.markdown('<div style="width:100%">', unsafe_allow_html=True)
            search_algo = st.selectbox("🔍 Search Algorithm", ["All", "DFS", "BFS", "UCS", "A*"], key="search")
            st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.model_choice = model_choice
    run_button = st.button("🚗 Find Route", type="primary")


# === Route Finding Logic ===
if run_button:
    travel_time_cache.clear()
    all_searches = search_algo == "All"
    search_fn_map = {"A*": astar, "BFS": bfs, "UCS": ucs, "DFS": dfs}
    search_methods = search_fn_map if all_searches else {search_algo: search_fn_map[search_algo]}
    st.session_state.results.clear()

    for name, search in search_methods.items():
        try:
            h_fn = (lambda n: heuristic_fn(n, destination)) if name == "A*" else None
            path, total_cost, segment_costs = search(origin, destination, get_neighbors, cost_fn, heuristic_fn=h_fn)

            if not path:
                st.session_state.results[name] = {"path": None, "error": "No path found."}
                continue

            rows = []
            cumulative = 0.0
            for i in range(1, len(path)):
                from_id, to_id = str(int(path[i - 1])), str(int(path[i]))
                key_str = (from_id, to_id)
                key_int = (int(from_id), int(to_id))
                delta = segment_costs.get(key_str) or segment_costs.get(key_int) or cost_fn(from_id, to_id)

                if delta == float("inf"):
                    continue

                cumulative += delta

                roads_from = set(metadata[from_id]["connected_roads"])
                roads_to = set(metadata[to_id]["connected_roads"])
                common = roads_from & roads_to
                road = sorted(common)[0] if common else "?"

                rows.append({
                    "From": from_id,
                    "To": to_id,
                    "Time (min)": round(delta, 2),
                    "Cumulative": round(cumulative, 2),
                    "Road": road
                })

            st.session_state.results[name] = {
                "path": [str(int(n)) for n in path],
                "total": cumulative,
                "table": rows
            }

        except Exception as e:
            st.session_state.results[name] = {"path": None, "error": str(e)}

colors = {"DFS": "black", "BFS": "purple", "UCS": "blue", "A*": "green"}
paths_for_map = {}

st.markdown("## 📈 Search Results")

for name, result in st.session_state.results.items():
    with st.container():
        st.subheader(f"🔍 {name} Result")
        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(f"<p style='font-size:18px;'><b>Route:</b> {' → '.join(result['path'])}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:16px;'><b>Estimated Travel Time:</b> {result['total']:.2f} minutes</p>", unsafe_allow_html=True)
            df = pd.DataFrame(result["table"])
            df.index = range(1, len(df) + 1)
            df.index.name = "Step"
            st.dataframe(df)

            paths_for_map[name] = result["path"]

if paths_for_map:
    st.markdown("## 🗺️ Visual Route Map")
    display_route_map(paths_for_map, metadata, colors)

import sys
st.write("🔍 Python path:", sys.executable)
st.write("🐍 Python version:", sys.version)
