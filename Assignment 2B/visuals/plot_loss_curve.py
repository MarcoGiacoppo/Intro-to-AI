import pandas as pd
import matplotlib.pyplot as plt
import os

models = ["lstm", "gru", "tcn"]
colors = {"lstm": "#007ACC", "gru": "#FF8C00", "tcn": "#8A2BE2"}

plt.figure(figsize=(12, 6))

for model in models:
    path = f"../results/loss_curve_{model}.csv"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    df = pd.read_csv(path)
    plt.plot(df["epoch"], df["loss"], label=f"{model.upper()} - Train", color=colors[model], linestyle="-")
    plt.plot(df["epoch"], df["val_loss"], label=f"{model.upper()} - Val", color=colors[model], linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

os.makedirs("../images", exist_ok=True)
plt.savefig("../images/loss_curve_all_models.png", dpi=300)
plt.show()
print("✅ Saved: ../images/loss_curve_all_models.png")
