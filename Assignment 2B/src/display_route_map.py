import folium
from streamlit_folium import st_folium

def display_route_map(paths: dict, metadata: dict, colors: dict):
    first_path = next(iter(paths.values()))
    start_info = metadata[str(first_path[0])]
    m = folium.Map(location=[start_info["latitude"], start_info["longitude"]], zoom_start=12)

    # --- Draw all SCATS sites with a slight shift
    for sid, info in metadata.items():
        lat = info["latitude"] + 0.0012
        lon = info["longitude"] + 0.0012
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.6,
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
            folium.PolyLine(coords, color=color, weight=5, opacity=0.7, popup=algo).add_to(m)

        # Start and end markers
        folium.Marker(coords[0], icon=folium.Icon(color="green", icon="play"), popup=f"{algo} Start").add_to(m)
        folium.Marker(coords[-1], icon=folium.Icon(color="red", icon="flag"), popup=f"{algo} End").add_to(m)

    return st_folium(m, width=700, height=500)
