import streamlit as st
import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model #type:ignore
import joblib
from search_algorithms import dfs, bfs, ucs, astar, gbfs, dijkstra
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

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
        background-color: #FAFAFA;
    }
    .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        margin: auto;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 0.92rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #E55750;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    .stDataFrameContainer {
        border-radius: 10px;
        overflow: hidden;
    }
    h1, h2, h3 {
        color: #222;
    }
    .param-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    .result-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #D5514B;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚦 Traffic-Based Route Guidance System</h1>", unsafe_allow_html=True)

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = {}

# === Input Section ===
st.markdown("<div class='param-header'>⚙️ Select Route Parameters</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox("🛫 Origin SCATS ID", site_ids, index=site_ids.index("970"), key="origin")
with col2:
    destination = st.selectbox("🏁 Destination SCATS ID", site_ids, index=site_ids.index("2000"), key="destination")

col3, col4 = st.columns(2)
with col3:
    model_choice = st.selectbox("🧠 Prediction Model", ["lstm", "gru", "tcn"], key="model")
with col4:
    search_algo = st.selectbox("🔍 Search Algorithm", ["All", "DFS", "BFS", "UCS", "Dijkstra", "GBFS", "A*"], key="search")

btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])
with btn_col3:
    run_button = st.button("🚗 Find Route")

st.session_state.model_choice = model_choice

# === Search Logic ===
if run_button:
    travel_time_cache.clear()
    all_searches = search_algo == "All"

    search_fn_map = {
        "A*": astar,
        "Dijkstra": dijkstra,
        "UCS": ucs,
        "BFS": bfs,
        "GBFS": gbfs,
        "DFS": dfs
    }

    search_methods = search_fn_map if all_searches else {search_algo: search_fn_map[search_algo]}
    st.session_state.results.clear()

    for name, search in search_methods.items():
        try:
            h_fn = (lambda n: heuristic_fn(n, destination)) if name in ["A*", "GBFS"] else None
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

# === Results Display ===
colors = {
<<<<<<< HEAD
    "A*": "#2ECC71",        # ✅ Green - Best
    "UCS": "#F1C40F",       # 🟡 Yellow - Slower but reliable
    "GBFS": "#E67E22",      # 🟠 Orange - Fast but not optimal
    "BFS": "#D35400",       # 🟠 Darker Orange
    "DFS": "#E74C3C"        # 🔴 Red - Worst
=======
    "A*": "#2ECC71",
    "Dijkstra": "#27AE60",
    "UCS": "#F1C40F",
    "GBFS": "#E67E22",
    "BFS": "#D35400",
    "DFS": "#E74C3C"
>>>>>>> 1c5727f6369fb5393248b528c67754334feb14a5
}

paths_for_map = {}

if st.session_state.results:
    st.markdown("---")
    st.markdown("<div class='param-header'>📈 Search Results</div>", unsafe_allow_html=True)

# Sort results by travel time (lowest first) and assign ranking
sorted_results = sorted(
    st.session_state.results.items(),
    key=lambda item: item[1]["total"] if item[1].get("total") else float("inf")
)

for idx, (name, result) in enumerate(sorted_results, start=1):
    with st.expander(f"Route {idx} -> {name}", expanded=False):
        st.markdown(f"<div class='result-header'>🔍 {name} Result</div>", unsafe_allow_html=True)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.markdown(f"<b>Route:</b> {' → '.join(result['path'])}", unsafe_allow_html=True)
            st.markdown(f"<b>Estimated Travel Time:</b> {result['total']:.2f} minutes", unsafe_allow_html=True)
            df = pd.DataFrame(result["table"])
            df.index = range(1, len(df) + 1)
            df.index.name = "Step"
            st.dataframe(df, use_container_width=True)
            paths_for_map[name] = result["path"]

# === Map Display ===
if paths_for_map:
    st.markdown("---")
    st.markdown("## 🗺️ Visual Route Map")
    st.info("Zoom and pan to explore the route paths.")
    display_route_map(paths_for_map, metadata, colors)
st.markdown("---")