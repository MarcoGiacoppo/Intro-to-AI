import pandas as pd
import json
import re
from collections import defaultdict
import os

# === Load data ===
xls_path = "../data/raw/Scats Data October 2006.xls"
df_data = pd.read_excel(xls_path, sheet_name="Data", header=1)
df_summary = pd.read_excel(xls_path, sheet_name="Summary Of Data", header=3)

# === Normalize SCATS numbers ===
df_data["SCATS Number"] = df_data["SCATS Number"].astype(str).str.zfill(4)
df_summary["SCATS Number"] = df_summary["SCATS Number"].fillna(method='ffill').astype(int).astype(str).str.zfill(4)

# === Group all location rows by SCATS site ===
site_locations = defaultdict(list)
for _, row in df_summary.iterrows():
    site_id = row["SCATS Number"]
    location = str(row["Location"]).strip()
    if location:
        site_locations[site_id].append(location)

# === Extract clean road names from location strings ===
def extract_roads(location):
    """Extract clean road names like HIGH STREET_RD, EASTERN_FWY_W_BD_RAMPS, etc."""
    if not isinstance(location, str):
        return []

    # Valid suffixes
    suffixes = ["RD", "ST", "HWY", "FWY", "GV", "ARTERIAL", "RAMPS", "PARK"]

    # Extract candidates: multi-word all-caps strings ending in a valid suffix
    road_pattern = re.compile(
        r"\b([A-Z0-9]+(?:[ _][A-Z0-9]+)*_(%s))\b" % "|".join(suffixes)
    )

    # Split on "of" (case-insensitive) to check both sides of the relation
    parts = re.split(r"\bof\b", location, flags=re.IGNORECASE)

    roads = set()
    for part in parts:
        for match in road_pattern.findall(part):
            full_name = match[0].strip()
            if full_name not in ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "OF"]:
                roads.add(full_name)

    return list(roads)


# === Construct metadata ===
site_metadata = {}

for site_id, locations in site_locations.items():
    roads = set()
    for loc in locations:
        roads.update(extract_roads(loc))

    # Get lat/lon from the first matching row in Data sheet
    lat, lon = None, None
    site_rows = df_data[df_data["SCATS Number"] == site_id]
    if not site_rows.empty:
            lat = lon = None
    for _, row in site_rows.iterrows():
        lat_candidate = row.get("NB_LATITUDE")
        lon_candidate = row.get("NB_LONGITUDE")
        if lat_candidate and lon_candidate and lat_candidate != 0 and lon_candidate != 0:
            lat = lat_candidate
            lon = lon_candidate
            break


    site_metadata[int(site_id)] = {
        "site_id": int(site_id),
        "latitude": float(lat) if lat else None,
        "longitude": float(lon) if lon else None,
        "locations": locations,
        "connected_roads": sorted(roads)
    }

# === Save to JSON ===
os.makedirs("../data/graph", exist_ok=True)
with open("../data/graph/sites_metadata.json", "w") as f:
    json.dump(site_metadata, f, indent=2)

print("✅ Metadata exported to ../data/graph/sites_metadata.json")
