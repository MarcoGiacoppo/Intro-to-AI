import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN

# === CONFIG ===
CSV_PATH = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"
MODEL_SAVE_PATH = "../models/tcn_model.h5"
RESULT_CSV = "../results/model_evaluation.csv"
SCATS_ID = "0970"
SEQ_LENGTH = 24
PRED_OFFSET = 1

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))
site_df = df[df["SCATS Number"] == SCATS_ID].sort_values("Date")

v_cols = [col for col in site_df.columns if col.startswith("V")]
values = site_df[v_cols].values.flatten()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values.reshape(-1, 1))

X, y = [], []
for i in range(len(scaled) - SEQ_LENGTH - PRED_OFFSET):
    X.append(scaled[i:i + SEQ_LENGTH])
    y.append(scaled[i + SEQ_LENGTH + PRED_OFFSET - 1])
X, y = np.array(X), np.array(y)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === MODEL ===
model = Sequential([
    TCN(input_shape=(SEQ_LENGTH, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)

# Save model
os.makedirs("../models", exist_ok=True)
model.save(MODEL_SAVE_PATH)

# === EVALUATION ===
y_pred = model.predict(X_test)
y_true = y_test

y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))

mae = mean_absolute_error(y_true_inv, y_pred_inv)
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)

# Append to results
os.makedirs("../results", exist_ok=True)
row = pd.DataFrame([{
    "model": "TCN",
    "SCATS_ID": SCATS_ID,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2
}])

if os.path.exists(RESULT_CSV):
    existing = pd.read_csv(RESULT_CSV)
    result = pd.concat([existing, row], ignore_index=True)
else:
    result = row

result.to_csv(RESULT_CSV, index=False)
print(row)
