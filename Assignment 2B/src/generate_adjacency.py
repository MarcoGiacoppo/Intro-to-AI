import json
import os
from collections import defaultdict
from math import radians, cos, sin, sqrt, atan2

# === Haversine utility ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# === Load metadata ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)

# === Inverted road index ===
road_to_sites = defaultdict(set)
for sid, info in metadata.items():
    for road in info["connected_roads"]:
        if road:
            road_to_sites[road].add(sid)

# === Build connections ===
adjacency = defaultdict(set)
MAX_DISTANCE = 3.0  # up to 3 km

for road, site_ids in road_to_sites.items():
    site_list = list(site_ids)
    for i in range(len(site_list)):
        for j in range(i + 1, len(site_list)):
            sid1, sid2 = site_list[i], site_list[j]
            site1, site2 = metadata[sid1], metadata[sid2]

            # Skip if missing location
            if not all([site1["latitude"], site1["longitude"], site2["latitude"], site2["longitude"]]):
                continue

            dist = haversine(site1["latitude"], site1["longitude"], site2["latitude"], site2["longitude"])
            if dist <= MAX_DISTANCE:
                adjacency[sid1].add(sid2)
                adjacency[sid2].add(sid1)

# === Final save ===
adjacency = {k: sorted(list(v)) for k, v in adjacency.items()}
os.makedirs("../data/graph", exist_ok=True)
with open("../data/graph/adjacency_from_summary.json", "w") as f:
    json.dump(adjacency, f, indent=2)

print(f"✅ adjacency_from_summary.json written with {len(adjacency)} nodes.")
