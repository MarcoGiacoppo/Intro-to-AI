import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Load flow CSVs
lstm = pd.read_csv("../results/flow_lstm.csv", parse_dates=["timestamp"])
gru  = pd.read_csv("../results/flow_gru.csv", parse_dates=["timestamp"])
tcn  = pd.read_csv("../results/flow_tcn.csv", parse_dates=["timestamp"])

def generate_heatmap(df, model_name):
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["abs_error"] = abs(df["true"] - df["predicted"])

    # === Trim to 30 days
    min_date = df["timestamp"].min().normalize()
    max_date = min_date + pd.Timedelta(days=30)
    df = df[(df["timestamp"] >= min_date) & (df["timestamp"] < max_date)]

    # === Pivot: rows = date, cols = hour
    pivot = df.pivot_table(index="date", columns="hour", values="abs_error", aggfunc="mean")

    # === Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="upper")

    ax.set_title(f"{model_name} – Avg Abs Error by Hour and Date")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Date")

    # Axis ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))

    # === Skip y-axis labels every 5 days
    step = 5
    visible_labels = [str(d) if i % step == 0 else "" for i, d in enumerate(pivot.index)]
    ax.set_yticklabels(visible_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(cax, label="Avg Abs Error")
    plt.tight_layout()

    # === Save
    os.makedirs("../images", exist_ok=True)
    save_path = f"../images/error_heatmap_{model_name.lower()}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

# === Generate heatmaps
generate_heatmap(lstm, "LSTM")
generate_heatmap(gru, "GRU")
generate_heatmap(tcn, "TCN")
