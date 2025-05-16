import json
import os
from collections import defaultdict
from routing_core import haversine

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

from operator import itemgetter

for road, site_ids in road_to_sites.items():
    site_list = []
    for sid in site_ids:
        info = metadata[sid]
        lat, lon = info["latitude"], info["longitude"]
        if lat is not None and lon is not None:
            site_list.append((sid, lat, lon))
    
    # Sort by approximate direction (latitude then longitude)
    site_list.sort(key=itemgetter(1, 2))  # sort by (lat, lon)

    # Connect only consecutive nodes
    for i in range(len(site_list) - 1):
        sid1, lat1, lon1 = site_list[i]
        sid2, lat2, lon2 = site_list[i + 1]

        dist = haversine(lat1, lon1, lat2, lon2)
        if dist <= MAX_DISTANCE:
            adjacency[sid1].add(sid2)
            adjacency[sid2].add(sid1)

# === Final save ===
adjacency = {k: sorted(list(v)) for k, v in adjacency.items()}
os.makedirs("../data/graph", exist_ok=True)
with open("../data/graph/adjacency_from_summary.json", "w") as f:
    json.dump(adjacency, f, indent=2)

print(f"✅ adjacency_from_summary.json written with {len(adjacency)} nodes.")
