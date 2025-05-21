import streamlit as st
import pandas as pd
from routing_core import (
    metadata, site_ids, travel_time_cache,
    cost_fn, heuristic_fn, get_neighbors, calculate_total_distance, preload_all_travel_times
)
from display_route_map import display_route_map
from search_algorithms import dfs, bfs, ucs, astar, gbfs, bidirectional

# === Page config and CSS ===
st.set_page_config(page_title="TBRGS - Route Finder", layout="wide")

st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #FAFAFA;
    }
    .block-container {
        max-width: 1150px;
        margin: auto;
    }
    .stSelectbox label, .stNumberInput label {
        font-size: 0.92rem;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #E55750;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
    }
    .stDataFrameContainer {
        border-radius: 10px;
        overflow: hidden;
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

# === App state ===
if "results" not in st.session_state:
    st.session_state.results = {}

# === Color schemes ===
colors = {
    "A*": "#00FF7F",
    "Bidirectional": "#1E90FF",
    "UCS": "#FFD700",
    "GBFS": "#FF69B4",
    "BFS": "#7B68EE",
    "DFS": "#FF4500",
}
color_names = {
    "A*": "Neon Green",
    "Bidirectional": "Dodger Blue",
    "UCS": "Gold",
    "GBFS": "Hot Pink",
    "BFS": "Slate Blue",
    "DFS": "Orange Red",
}

# === Input section ===
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
    search_algo = st.selectbox("🔍 Search Algorithm", ["All", "DFS", "BFS", "UCS", "Bidirectional", "GBFS", "A*"], key="search")

# new addition, hour of day
col5, _ = st.columns([1, 1])
with col5:
    selected_hour = st.slider("🕒 Hour of Day (0–23)", min_value=0, max_value=23, value=8, step=1, key="hour")

# 👇 DEBUG: Show selected parameters
print(f"[DEBUG][GUI] Selected model: {model_choice} | Hour: {selected_hour}")

btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1])
with btn_col3:
    run_button = st.button("🚗 Find Route")

paths_for_map = {}

# === Search execution ===
if run_button:
    travel_time_cache.clear()

    all_searches = search_algo == "All"

    search_fn_map = {
        "A*": astar,
        "Bidirectional": bidirectional,
        "UCS": ucs,
        "BFS": bfs,
        "GBFS": gbfs,
        "DFS": dfs
    }

    search_methods = search_fn_map if all_searches else {search_algo: search_fn_map[search_algo]}
    st.session_state.results.clear()

    travel_time_cache.clear()

    travel_time_cache.update(
        preload_all_travel_times(model_choice, selected_hour)
    )


    for name, search in search_methods.items():
        try:
            h_fn = (lambda n: heuristic_fn(n, destination)) if name in ["A*", "GBFS"] else None
            path, total_cost, segment_costs = search(
                origin, 
                destination, 
                get_neighbors, 
                lambda a, b: cost_fn(a, b, model_choice, selected_hour), 
                heuristic_fn=h_fn)

            if not path:
                st.session_state.results[name] = {"path": None, "error": "No path found."}
                continue

            rows, cumulative = [], 0.0
            for i in range(1, len(path)):
                from_id, to_id = str(int(path[i - 1])), str(int(path[i]))
                key_str = (from_id, to_id)
                delta = segment_costs.get(key_str)
                if delta is None:
                    delta = "?"

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

# === Summary and results ===
if st.session_state.results:
    sorted_results = sorted(
        st.session_state.results.items(),
        key=lambda item: item[1]["total"] if item[1].get("total") else float("inf")
    )

    summary_data = []
    for idx, (name, result) in enumerate(sorted_results, start=1):
        if result.get("total") and result.get("path"):
            total_km = calculate_total_distance(result["path"])
            summary_data.append({
                "Algorithm": name,
                "Estimated Time (min)": round(result["total"], 2),
                "Total Distance (km)": total_km,
                "Steps": len(result["path"]) - 1,
                "Heuristic Used?": "Yes" if name in ["A*", "GBFS"] else "No",
                "Color": color_names.get(name, "N/A")
            })

    if summary_data:
        st.markdown("### 📋 Route Summary (Ranked by Estimated Time)")
        summary_df = pd.DataFrame(summary_data)
        summary_df.index = [f"Route {i}" for i in range(1, len(summary_df) + 1)]
        summary_df.index.name = "Route"
        st.dataframe(summary_df, use_container_width=True)

    # Results expander
    st.markdown("---")
    st.markdown("### 📈 Search Results")

    for idx, (name, result) in enumerate(sorted_results, start=1):
        algo_color_name = color_names.get(name, "Unknown")
        with st.expander(f"Route {idx} → {name} ({algo_color_name})", expanded=False):
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

    # === Map display ===
    if paths_for_map:
        st.markdown("---")
        st.markdown("## 🗺️ Visual Route Map")
        st.info("Zoom and pan to explore the route paths.")
        display_route_map(paths_for_map, metadata, colors)

    st.markdown("---")