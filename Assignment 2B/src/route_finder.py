import json
import math
from predict_travel_time import predict_travel_time
import search_algorithms

# === Load data ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)
with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

# === Utility functions ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_neighbors(node):
    return adjacency.get(str(node).zfill(4), [])

def cost_fn(a, b):
    a = str(a).zfill(4)
    b = str(b).zfill(4)
    if a not in metadata or b not in metadata:
        return float("inf")
    m1, m2 = metadata[a], metadata[b]
    if not all([m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"]]):
        return float("inf")
    dist = haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])
    travel_time = predict_travel_time(a)
    return dist * travel_time

def heuristic_fn(n):
    n = str(n).zfill(4)
    if n not in metadata or goal not in metadata:
        return 0
    m1 = metadata[n]
    m2 = metadata[goal]
    if not all([m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"]]):
        return 0
    return haversine(m1["latitude"], m1["longitude"], m2["latitude"], m2["longitude"])

# === CLI Interface ===
if __name__ == "__main__":
    print("🛣️  TBRGS: Route Finder with Travel-Time Estimation")

    start = input("Enter origin SCATS number: ").strip().zfill(4)
    goal = input("Enter destination SCATS number: ").strip().zfill(4)

    if start not in metadata or goal not in metadata:
        print("🚫 Invalid SCATS site ID(s).")
        exit()

    print("\nAvailable search methods: dfs, bfs, ucs, gbfs, astar")
    method = input("Choose search method: ").strip().lower()

    if method == "dfs":
        path, total_cost, segment_costs = search_algorithms.dfs(start, goal, get_neighbors, cost_fn, heuristic_fn)
    elif method == "bfs":
        path, total_cost, segment_costs = search_algorithms.bfs(start, goal, get_neighbors, cost_fn, heuristic_fn)
    elif method == "ucs":
        path, total_cost, segment_costs = search_algorithms.ucs(start, goal, get_neighbors, cost_fn, heuristic_fn)
    elif method == "gbfs":
        path, total_cost, segment_costs = search_algorithms.gbfs(start, goal, get_neighbors, cost_fn, heuristic_fn)
    elif method == "astar":
        path, total_cost, segment_costs = search_algorithms.astar(start, goal, get_neighbors, cost_fn, heuristic_fn)
    else:
        print("❌ Invalid search method.")
        exit()

    if not path:
        print("❌ No path found.")
        exit()

    # Print path summary
    print(f"\n✅ Path from {start} to {goal}")
    print(f"🕒 Estimated total travel time: {round(total_cost, 2)} min")
    print(f"{'Step':<5} {'From':<6} {'To':<6} {'Time (min)':<12} {'Road'}")
    print("-" * 60)

    cumulative = 0.0
    for i in range(1, len(path)):
        from_id, to_id = path[i - 1], path[i]
        delta = segment_costs.get((from_id, to_id))
        if delta is None:
            delta = cost_fn(from_id, to_id)
        cumulative += delta
        roads_from = set(metadata[from_id]["connected_roads"])
        roads_to = set(metadata[to_id]["connected_roads"])
        common = roads_from & roads_to
        road = sorted(common)[0] if common else "?"
        print(f"{i:<5} {from_id:<6} {to_id:<6} {delta:<12.2f} {road}")

    # Ask for visualization
    visualize = input("Would you like to visualize the route? (y/n): ").strip().lower()
    if visualize == "y":
        import visualize_route
        visualize_route.plot_graph_and_route(path, metadata, adjacency)
