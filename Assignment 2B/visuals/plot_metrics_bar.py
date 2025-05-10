import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load evaluation metrics CSV ===
df = pd.read_csv("../results/model_evaluation.csv")

# === Metrics to show ===
metrics = ["MAE", "MSE", "RMSE", "R2"]
x = range(len(metrics))
bar_width = 0.25

# === Pull model scores ===
lstm = df.loc[df["model"] == "LSTM", metrics].values[0]
gru = df.loc[df["model"] == "GRU", metrics].values[0]
tcn = df.loc[df["model"] == "TCN", metrics].values[0]

# === Plot chart ===
plt.figure(figsize=(10, 6))
plt.bar([p - bar_width for p in x], lstm, bar_width, label="LSTM")
plt.bar(x, gru, bar_width, label="GRU")
plt.bar([p + bar_width for p in x], tcn, bar_width, label="TCN")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Model Evaluation Metrics (Lower is Better, Except R²)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# === Save chart ===
os.makedirs("../images", exist_ok=True)
save_path = "../images/metrics_comparison.png"
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Chart saved to: {save_path}")
