import folium
from streamlit_folium import st_folium

def display_route_map(paths: dict, metadata: dict, colors: dict):
    # Get the first node to center the map
    first_path = next(iter(paths.values()))
    start_info = metadata[first_path[0]]
    m = folium.Map(location=[start_info["latitude"], start_info["longitude"]], zoom_start=12)

    # Plot all SCATS sites
    for sid, info in metadata.items():
        sid = str(sid)
        folium.CircleMarker(
            location=[info["latitude"], info["longitude"]],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.5,
            popup=f"SCATS: {sid}"
        ).add_to(m)

    # Draw each path with a different color
    for label, path in paths.items():
        route_coords = [[metadata[str(sid)]["latitude"], metadata[str(sid)]["longitude"]] for sid in path]
        folium.PolyLine(
            route_coords,
            color=colors.get(label, "blue"),
            weight=5,
            opacity=0.8,
            tooltip=f"{label} path"
        ).add_to(m)

        # Start and end markers
        folium.Marker(route_coords[0], icon=folium.Icon(color="green", icon="play"), popup=f"{label} Start").add_to(m)
        folium.Marker(route_coords[-1], icon=folium.Icon(color="red", icon="flag"), popup=f"{label} End").add_to(m)

    # Show the map in Streamlit
    st_folium(m, width=900, height=600)
