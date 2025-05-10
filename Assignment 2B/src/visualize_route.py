import matplotlib.pyplot as plt

def plot_graph_and_route(path, metadata, adjacency):
    plt.figure(figsize=(12, 12))
    plt.title("SCATS Site Network — Boroondara")

    # === Plot all nodes and edges ===
    for site_id, neighbors in adjacency.items():
        site = metadata.get(site_id)
        if not site: continue

        lat1, lon1 = site["latitude"], site["longitude"]
        for neighbor in neighbors:
            neighbor_site = metadata.get(neighbor)
            if not neighbor_site: continue
            lat2, lon2 = neighbor_site["latitude"], neighbor_site["longitude"]
            plt.plot([lon1, lon2], [lat1, lat2], color="lightgrey", linewidth=0.7, zorder=1)

    # === Plot all SCATS sites ===
    for sid, info in metadata.items():
        plt.scatter(info["longitude"], info["latitude"], color="violet", s=15, zorder=2)
        plt.text(info["longitude"], info["latitude"], sid, fontsize=7, ha='center', va='bottom', color="black")

    # === Highlight path ===
    for i in range(len(path) - 1):
        sid1, sid2 = path[i], path[i+1]
        n1, n2 = metadata[sid1], metadata[sid2]
        plt.plot([n1["longitude"], n2["longitude"]], [n1["latitude"], n2["latitude"]],
                 color="blue", linewidth=2.5, zorder=3)

    # === Highlight start and end ===
    start = metadata[path[0]]
    end = metadata[path[-1]]
    plt.scatter(start["longitude"], start["latitude"], color="green", s=80, label="Start", zorder=4)
    plt.scatter(end["longitude"], end["latitude"], color="red", s=80, label="End", zorder=4)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
