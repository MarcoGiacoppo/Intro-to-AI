#============================================
## Still needs to be fixed. very bad for now
#============================================

import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from search_algorithms import dfs, bfs, ucs, astar
from visualize_route import plot_graph_and_route

# === Load metadata and adjacency ===
with open("../data/graph/sites_metadata.json") as f:
    metadata = json.load(f)
with open("../data/graph/adjacency_from_summary.json") as f:
    adjacency = json.load(f)

# === Helper functions ===
def get_neighbors(node):
    return adjacency.get(str(node), [])

def predict_travel_time_model(scats_id, model_name="lstm"):
    scats_id = str(scats_id).zfill(4)
    hour = datetime.now().hour

    model_path = f"../models/{model_name}_model.keras"
    scaler_path = f"../models/{model_name}_scaler.pkl"
    encoder_path = f"../models/{model_name}_scats_encoder.pkl"

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    if scats_id not in encoder.classes_:
        return 1.0

    idx = encoder.transform([scats_id])[0]
    dummy_seq = np.full((1, 24, 1), 0.5)
    input_scats = np.array([[idx]])

    pred_scaled = model.predict([dummy_seq, input_scats], verbose=0)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]

    a, b = 0.001, 1.6
    speed = max(5.0, 100 * (1 - a * (pred_volume ** b)))
    return round(60 / speed, 2)

def cost_fn(a, b):
    return predict_travel_time_model(b, model_var.get())

def run_search():
    origin = origin_entry.get().strip()
    dest = dest_entry.get().strip()
    model = model_var.get()
    algo = algo_var.get()

    if not origin or not dest:
        messagebox.showerror("Missing Input", "Enter both origin and destination.")
        return

    try:
        search_methods = {"DFS": dfs, "BFS": bfs, "UCS": ucs, "A*": astar}
        search = search_methods[algo]
        path, cost, _ = search(origin, dest, get_neighbors, cost_fn, heuristic_fn=None)

        if not path:
            result_box.delete("1.0", tk.END)
            result_box.insert(tk.END, "No route found.\n")
            return

        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, f"{algo} result using {model}:\n")
        result_box.insert(tk.END, f"Path: {' -> '.join(path)}\n")
        result_box.insert(tk.END, f"Estimated travel time: {cost:.2f} min\n")

        plot_graph_and_route(path, metadata, adjacency)

    except Exception as e:
        messagebox.showerror("Error", str(e))

# === GUI Setup ===
root = tk.Tk()
root.title("TBRGS - Route Finder")

# Origin
tk.Label(root, text="Origin SCATS ID:").grid(row=0, column=0, sticky="e")
origin_entry = tk.Entry(root)
origin_entry.grid(row=0, column=1, pady=5)

# Destination
tk.Label(root, text="Destination SCATS ID:").grid(row=1, column=0, sticky="e")
dest_entry = tk.Entry(root)
dest_entry.grid(row=1, column=1, pady=5)

# Model dropdown
tk.Label(root, text="Model:").grid(row=2, column=0, sticky="e")
model_var = tk.StringVar(value="lstm")
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["lstm", "gru", "tcn"], state="readonly")
model_dropdown.grid(row=2, column=1, pady=5)

# Algorithm dropdown
tk.Label(root, text="Search Algorithm:").grid(row=3, column=0, sticky="e")
algo_var = tk.StringVar(value="UCS")
algo_dropdown = ttk.Combobox(root, textvariable=algo_var, values=["DFS", "BFS", "UCS", "A*"], state="readonly")
algo_dropdown.grid(row=3, column=1, pady=5)

# Button
tk.Button(root, text="Find Route", command=run_search).grid(row=4, column=0, columnspan=2, pady=10)

# Results
result_box = tk.Text(root, width=60, height=10)
result_box.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
