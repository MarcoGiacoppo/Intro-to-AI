import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# === CONFIG ===
CSV_PATH = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"
MODEL_SAVE_PATH = "../models/lstm_model.h5"
SCALER_PATH = "../models/scaler.pkl"
TARGET_SCALER_PATH = "../models/target_scaler.pkl"
RESULT_CSV = "../results/model_evaluation.csv"

# === Load and Prepare Data ===
df = pd.read_csv(CSV_PATH)
df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))

# Extract 15-min bins V00–V95 and melt to long format
v_cols = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]
df_long = df.melt(id_vars=["SCATS Number", "Date"], value_vars=v_cols,
                  var_name="Bin", value_name="Volume")

# Derive hour from Bin (V00 → 0:00, V01 → 0:15, ..., V95 → 23:45)
df_long["hour"] = df_long["Bin"].apply(lambda x: int(x[1:]) // 4)

# Group by SCATS Number and hour, then average the volume
grouped = df_long.groupby(["SCATS Number", "hour"])["Volume"].mean().reset_index()

# Encode inputs and targets
X_raw = grouped[["SCATS Number", "hour"]].astype(int).values
y_raw = grouped["Volume"].values.reshape(-1, 1)

# Scale
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# Save scalers
os.makedirs("../models", exist_ok=True)
joblib.dump(scaler_X, SCALER_PATH)
joblib.dump(scaler_y, TARGET_SCALER_PATH)

# Train-test split
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# Reshape for LSTM (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# === Build and Train LSTM ===
model = Sequential([
    LSTM(64, input_shape=(1, 2)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=[es], verbose=1)

# Save model
model.save(MODEL_SAVE_PATH)

# === Evaluate ===
y_pred = model.predict(X_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_true_inv = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true_inv, y_pred_inv)
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true_inv - y_pred_inv) / np.maximum(y_true_inv, 1e-5))) * 100
r2 = r2_score(y_true_inv, y_pred_inv)

# Save results
os.makedirs("../results", exist_ok=True)
eval_df = pd.DataFrame([{
    "model": "LSTM",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "MAPE": mape,
    "R2": r2
}])
eval_df.to_csv(RESULT_CSV, index=False)
print(eval_df)
