import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# === Load patched CSVs ===
lstm = pd.read_csv("../results/flow_lstm.csv", parse_dates=["timestamp"])
gru = pd.read_csv("../results/flow_gru.csv", parse_dates=["timestamp"])
tcn = pd.read_csv("../results/flow_tcn.csv", parse_dates=["timestamp"])

# === Convert to datetime index for resampling
lstm.set_index("timestamp", inplace=True)
gru.set_index("timestamp", inplace=True)
tcn.set_index("timestamp", inplace=True)

# === Average across all SCATS sites per timestamp
lstm_avg = lstm.groupby("timestamp").mean(numeric_only=True)
gru_avg  = gru.groupby("timestamp").mean(numeric_only=True)
tcn_avg  = tcn.groupby("timestamp").mean(numeric_only=True)

# === Define date range to visualize (first full day)
start_date = lstm_avg.index.min().normalize()
end_date = start_date + pd.Timedelta(days=1)
mask = (lstm_avg.index >= start_date) & (lstm_avg.index < end_date)

# === Extract values
timestamps = lstm_avg.loc[mask].index
true_vals  = lstm_avg.loc[mask]["true"]
lstm_vals  = lstm_avg.loc[mask]["predicted"]
gru_vals   = gru_avg.loc[mask]["predicted"]
tcn_vals   = tcn_avg.loc[mask]["predicted"]

# === Plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, true_vals, label="True Volume (Avg)", color="black", linewidth=2)
plt.plot(timestamps, lstm_vals, "--", label="LSTM (Avg)", color="#007ACC")
plt.plot(timestamps, gru_vals, "--", label="GRU (Avg)", color="#FF8C00")
plt.plot(timestamps, tcn_vals, "--", label="TCN (Avg)", color="#8A2BE2")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)

plt.xlabel("Time of Day")
plt.ylabel("Average Traffic Volume")
plt.title("True vs Predicted Traffic Flow (Avg across sites)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# === Save image
os.makedirs("../images", exist_ok=True)
plt.savefig("../images/flow_time_series_comparison_avg.png", dpi=300)
plt.show()

print("✅ Chart saved to: /images/flow_time_series_comparison_avg.png")
