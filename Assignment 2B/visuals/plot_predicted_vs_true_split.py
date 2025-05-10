import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load flow prediction CSVs
lstm = pd.read_csv("../results/flow_lstm.csv")
gru  = pd.read_csv("../results/flow_gru.csv")
tcn  = pd.read_csv("../results/flow_tcn.csv")

# === Extract true values (same for all)
true_vals = lstm["true"]

# === Shared plot settings
max_val = max(true_vals.max(), lstm["predicted"].max(), gru["predicted"].max(), tcn["predicted"].max())

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
models = [("LSTM", lstm), ("GRU", gru), ("TCN", tcn)]
colors = ["#007ACC", "#FF8C00", "#8A2BE2"]

for ax, (name, df), color in zip(axs, models, colors):
    ax.scatter(true_vals, df["predicted"], alpha=0.3, s=10, color=color)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="black")
    ax.set_title(f"{name} Prediction vs True")
    ax.set_xlabel("True Flow")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle="--", alpha=0.4)

axs[0].set_ylabel("Predicted Flow")
plt.suptitle("Predicted vs True Traffic Flow by Model", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# === Save
os.makedirs("../images", exist_ok=True)
save_path = "../images/predicted_vs_true_split.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved to: {save_path}")
