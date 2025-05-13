import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load flow files
lstm = pd.read_csv("../results/flow_lstm.csv", parse_dates=["timestamp"])
gru = pd.read_csv("../results/flow_gru.csv", parse_dates=["timestamp"])
tcn = pd.read_csv("../results/flow_tcn.csv", parse_dates=["timestamp"])

# === Compute absolute error
lstm["error"] = abs(lstm["true"] - lstm["predicted"])
gru["error"] = abs(gru["true"] - gru["predicted"])
tcn["error"] = abs(tcn["true"] - tcn["predicted"])

# === Group by date
lstm_daily = lstm.groupby(lstm["timestamp"].dt.date)["error"].mean()
gru_daily = gru.groupby(gru["timestamp"].dt.date)["error"].mean()
tcn_daily = tcn.groupby(tcn["timestamp"].dt.date)["error"].mean()

# === Plot
plt.figure(figsize=(12, 6))
plt.plot(lstm_daily.index, lstm_daily.values, label="LSTM", color="#007ACC", linewidth=2)
plt.plot(gru_daily.index, gru_daily.values, label="GRU", color="#FF8C00", linewidth=2)
plt.plot(tcn_daily.index, tcn_daily.values, label="TCN", color="#8A2BE2", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Avg Absolute Error")
plt.title("Prediction Error Over Time")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# === Save
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/error_over_time.png", dpi=300)
plt.close()
print("✅ Saved: ../images/error_over_time.png")
