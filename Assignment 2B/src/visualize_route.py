import json
import matplotlib.pyplot as plt
from route_finder import a_star_detailed

# === Load metadata and adjacency ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)

with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

# === Utility to get lat/lon ===
def get_coords(site_id):
    site = metadata.get(site_id.zfill(4))
    if site and site["latitude"] and site["longitude"]:
        return float(site["longitude"]), float(site["latitude"])
    return None, None

# === Function to visualize the network and a given route ===
def visualize_route(route=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("SCATS Site Network (Boroondara)", fontsize=16)

    # Plot edges
    for src, neighbors in adjacency.items():
        for tgt in neighbors:
            x1, y1 = get_coords(src)
            x2, y2 = get_coords(tgt)
            if None not in (x1, y1, x2, y2):
                ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)

    # Plot nodes
    for sid, data in metadata.items():
        if data["latitude"] and data["longitude"]:
            x, y = float(data["longitude"]), float(data["latitude"])
            ax.scatter(x, y, color='blue', s=15)
            ax.text(x, y, sid, fontsize=6, ha='right', va='bottom')

    # Highlight route if provided
    if route:
        route_x, route_y = [], []
        for sid in route:
            x, y = get_coords(sid)
            if None not in (x, y):
                route_x.append(x)
                route_y.append(y)
        ax.plot(route_x, route_y, color='red', linewidth=2.5, label="Route")
        ax.scatter(route_x, route_y, color='red', s=30)
        ax.legend()

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Dynamic input + route finding ===
if __name__ == "__main__":
    print("📍 SCATS Visual Route Finder")
    origin = input("Enter origin SCATS number: ").strip().zfill(4)
    destination = input("Enter destination SCATS number: ").strip().zfill(4)

    path, _ = a_star_detailed(origin, destination)

    if not path:
        print("❌ No route found.")
    else:
        print(f"✅ Visualizing route from {origin} to {destination}...")
        visualize_route(path)