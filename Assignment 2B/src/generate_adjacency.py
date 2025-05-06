import json
import os
from collections import defaultdict

# === Load metadata ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)

# === Build inverted index: road name → list of SCATS IDs ===
road_to_sites = defaultdict(set)

for scats_id, info in metadata.items():
    for road in info["connected_roads"]:
        road_to_sites[road].add(scats_id)

# === Create adjacency map ===
adjacency = defaultdict(set)

for sites in road_to_sites.values():
    site_list = list(sites)
    for i in range(len(site_list)):
        for j in range(len(site_list)):
            if i != j:
                adjacency[site_list[i]].add(site_list[j])

# Convert to JSON-serializable format
adjacency = {k: sorted(list(v)) for k, v in adjacency.items()}

# === Save it ===
os.makedirs("../data/graph", exist_ok=True)
with open("../data/graph/adjacency_from_summary.json", "w") as f:
    json.dump(adjacency, f, indent=2)

print("✅ adjacency_from_summary.json created successfully.")
