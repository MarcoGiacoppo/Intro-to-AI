import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load results ===
df = pd.read_csv("../results/model_evaluation.csv")
metrics = ["MAE", "RMSE", "R2"]

# === Bar Plot ===
plt.figure(figsize=(10, 6))
bar_width = 0.25
x = range(len(df["model"]))

for i, metric in enumerate(metrics):
    plt.bar([p + i * bar_width for p in x], df[metric], width=bar_width, label=metric)

plt.xticks([p + bar_width for p in x], df["model"])
plt.ylabel("Metric Value")
plt.title("Model Evaluation Metrics Comparison")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

os.makedirs("../images", exist_ok=True)
plt.tight_layout()
plt.savefig("../images/model_metric_comparison.png", dpi=300)
plt.show()

print("✅ Metric comparison plot saved to: /images/model_metric_comparison.png")
