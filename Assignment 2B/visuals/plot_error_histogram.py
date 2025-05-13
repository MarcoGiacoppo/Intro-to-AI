import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# === Configuration ===
models = ["lstm", "gru", "tcn"]
colors = {"lstm": "#007ACC", "gru": "#FF8C00", "tcn": "#8A2BE2"}
labels = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN"}

plt.figure(figsize=(10, 6))

for model in models:
    path = f"../results/flow_{model}.csv"
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        continue

    df = pd.read_csv(path)
    df["abs_error"] = abs(df["true"] - df["predicted"])

    # Plot as smoothed step histogram
    counts, bins = np.histogram(df["abs_error"], bins=40, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, counts, label=labels[model], color=colors[model], linewidth=2)

plt.title("Distribution of Absolute Errors", fontsize=14)
plt.xlabel("Absolute Error", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

os.makedirs("../images", exist_ok=True)
plt.savefig("../images/histogram_abs_error.png", dpi=300)
plt.show()

print("✅ Saved: ../images/histogram_abs_error.png")
