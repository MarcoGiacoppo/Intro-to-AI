import json
import math
import heapq
from predict_travel_time import predict_travel_time

# === Load files ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)
with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

# === Haversine formula ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# === A* Search with detailed step tracking ===
def a_star_detailed(start, goal):
    start = str(start).zfill(4)
    goal = str(goal).zfill(4)

    if start not in metadata or goal not in metadata:
        print("🚫 Invalid SCATS site ID(s).")
        return [], []

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    segment_times = {}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for neighbor in adjacency.get(current, []):
            cur_data = metadata.get(current)
            next_data = metadata.get(neighbor)

            if not all([cur_data["latitude"], cur_data["longitude"], next_data["latitude"], next_data["longitude"]]):
                continue

            dist = haversine(cur_data["latitude"], cur_data["longitude"],
                             next_data["latitude"], next_data["longitude"])
            travel_time_per_km = predict_travel_time(current)
            travel_time = dist * travel_time_per_km


            new_cost = cost_so_far[current] + travel_time

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                segment_times[(current, neighbor)] = travel_time
                priority = new_cost + haversine(next_data["latitude"], next_data["longitude"],
                                                metadata[goal]["latitude"], metadata[goal]["longitude"])
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return [], []

    # Reconstruct path and times
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()

    cumulative = 0
    times = []
    for i in range(len(path)):
        if i == 0:
            times.append(0.0)
        else:
            delta = segment_times.get((path[i-1], path[i]), 0)
            cumulative += delta
            times.append(round(cumulative, 2))

    return path, times

# === Run CLI ===
if __name__ == "__main__":
    print("🛣️  TBRGS: Route Finder with Detailed Times")
    start = input("Enter origin SCATS number: ").strip()
    goal = input("Enter destination SCATS number: ").strip()

    path, time_steps = a_star_detailed(start, goal)

    if not path:
        print("❌ No path found.")
    else:
        print(f"\n✅ Path from {start.zfill(4)} to {goal.zfill(4)} — Total time: {time_steps[-1]} min\n")
        print(f"{'Step':<5} {'SCATS':<8} {'Time (min)':<12} Roads")
        print("-" * 60)
        for i, sid in enumerate(path):
            roads = ", ".join(metadata[sid]["connected_roads"])
            print(f"{i+1:<5} {sid:<8} {time_steps[i]:<12.2f} {roads}")
