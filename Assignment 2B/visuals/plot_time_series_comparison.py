import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# === Load patched CSVs ===
lstm = pd.read_csv("../results/flow_lstm.csv", parse_dates=["timestamp"])
gru = pd.read_csv("../results/flow_gru.csv", parse_dates=["timestamp"])
tcn = pd.read_csv("../results/flow_tcn.csv", parse_dates=["timestamp"])

# === Automatically find first day range ===
first_timestamp = lstm["timestamp"].min()
start_date = first_timestamp.normalize()
end_date = start_date + pd.Timedelta(days=1) # Change days value to show combined data for that amount of days

# === Filter data ===
mask = (lstm["timestamp"] >= start_date) & (lstm["timestamp"] < end_date)
timestamps = lstm.loc[mask, "timestamp"]
true_vals = lstm.loc[mask, "true"]
lstm_vals = lstm.loc[mask, "predicted"]
gru_vals = gru.loc[mask, "predicted"]
tcn_vals = tcn.loc[mask, "predicted"]

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(timestamps, true_vals, label="True Data", color="black", linewidth=2)
plt.plot(timestamps, lstm_vals, "--", label="LSTM", color="#007ACC")
plt.plot(timestamps, gru_vals, "--", label="GRU", color="#FF8C00")
plt.plot(timestamps, tcn_vals, "--", label="TCN", color="#8A2BE2")

# === Format x-axis to show time only ===
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

plt.xlabel("Time of Day")
plt.ylabel("Traffic Flow")
plt.title("True vs Predicted Traffic Flow")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# === Save image ===
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/flow_time_series_comparison.png", dpi=300)
plt.show()

print("✅ Chart saved to: /images/flow_time_series_comparison.png")