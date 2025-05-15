import folium
from streamlit_folium import st_folium
from folium.plugins import AntPath

def display_route_map(paths: dict, metadata: dict, colors: dict):
    first_path = next(iter(paths.values()))
    start_info = metadata[str(first_path[0])]
    adjusted_lat = start_info["latitude"] + 0.04  # move view 
    adjusted_lon = start_info["longitude"] - 0.02
    m = folium.Map(location=[adjusted_lat, adjusted_lon], zoom_start=13, tiles="CartoDB dark_matter") #remove tiles if want to change back to default


    # Gather all node IDs involved in any route
    highlighted_nodes = set(str(sid) for path in paths.values() for sid in path)

    # Render all nodes, coloring them conditionally
    for sid, info in metadata.items():
        lat = info["latitude"] + 0.0012
        lon = info["longitude"] + 0.0012
        is_in_path = str(sid) in highlighted_nodes

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="red" if is_in_path else "blue",
            fill=True,
            fill_opacity=0.8 if is_in_path else 0.3,
            popup=f"SCATS: {sid}"
        ).add_to(m)


    # --- Draw each route
    for algo, path in paths.items():
        color = colors.get(algo, "blue")
        coords = []

        for sid in path:
            sid = str(sid)
            info = metadata.get(sid)
            if info:
                lat = info["latitude"] + 0.0012
                lon = info["longitude"] + 0.0012
                coords.append((lat, lon))

        if coords:
            AntPath(
                locations=coords,
                color=color,
                weight=5,
                delay=800,  # animation speed in ms
                dash_array=[10, 20],
                pulse_color="#fff",
            ).add_to(m)


        # Start and end markers
        folium.Marker(coords[0], icon=folium.Icon(color="green", icon="play"), popup=f"{algo} Start").add_to(m)
        folium.Marker(coords[-1], icon=folium.Icon(color="red", icon="flag"), popup=f"{algo} End").add_to(m)

    return st_folium(m, width=1000)