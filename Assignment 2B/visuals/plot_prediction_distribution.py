import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load predicted values
lstm = pd.read_csv("../results/flow_lstm.csv")
gru = pd.read_csv("../results/flow_gru.csv")
tcn = pd.read_csv("../results/flow_tcn.csv")

# === Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(lstm["predicted"], bins=50, alpha=0.6, label="LSTM", color="#007ACC")
plt.hist(gru["predicted"], bins=50, alpha=0.6, label="GRU", color="#FF8C00")
plt.hist(tcn["predicted"], bins=50, alpha=0.6, label="TCN", color="#8A2BE2")

plt.xlabel("Predicted Traffic Volume")
plt.ylabel("Frequency")
plt.title("Prediction Distribution by Model")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# === Save
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/prediction_distribution.png", dpi=300)
plt.close()
print("✅ Saved: ../images/prediction_distribution.png")
