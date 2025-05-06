import pandas as pd
import json
import re
from collections import defaultdict
import os

# === Load data ===
xls_path = "../data/Scats Data October 2006.xls"
df_data = pd.read_excel(xls_path, sheet_name="Data", header=1)
df_summary = pd.read_excel(xls_path, sheet_name="Summary Of Data", header=3)

# === Clean SCATS Number column ===
df_data["SCATS Number"] = df_data["SCATS Number"].astype(str).str.zfill(4)
df_summary["SCATS Number"] = df_summary["SCATS Number"].fillna(method='ffill').astype(int).astype(str).str.zfill(4)

# === Group locations by site ===
site_locations = defaultdict(list)
for _, row in df_summary.iterrows():
    scats_id = row["SCATS Number"]
    location = row["Location"]
    if isinstance(location, str):
        site_locations[scats_id].append(location)

# === Extract roads ===
def extract_roads(location):
    if not isinstance(location, str):
        return []

    # Remove directional components like "E of", "N of", etc.
    cleaned = re.sub(r"\b(?:N|S|E|W|NE|NW|SE|SW)\b\s+of\b", "", location)
    cleaned = re.sub(r"\bof\b", "", cleaned)

    # Extract uppercase road-like tokens (e.g., WARRIGAL_RD, HIGH_ST, BURWOOD_HWY)
    matches = re.findall(r"[A-Z]+(?:_[A-Z]+)+", cleaned)

    return matches

site_metadata = {}

for site_id, locations in site_locations.items():
    roads = set()
    for loc in locations:
        roads.update(extract_roads(loc))

    lat, lon = None, None
    site_data_rows = df_data[df_data["SCATS Number"] == site_id]
    if not site_data_rows.empty:
        lat = site_data_rows["NB_LATITUDE"].iloc[0]
        lon = site_data_rows["NB_LONGITUDE"].iloc[0]

    site_metadata[site_id] = {
        "site_id": site_id,
        "latitude": float(lat) if lat else None,
        "longitude": float(lon) if lon else None,
        "locations": locations,
        "connected_roads": sorted(roads),
    }

# === Save to JSON ===
os.makedirs("../data/graph", exist_ok=True)
with open("../data/graph/sites_metadata.json", "w") as f:
    json.dump(site_metadata, f, indent=2)

print("✅ Metadata exported to ../data/graph/sites_metadata.json")
