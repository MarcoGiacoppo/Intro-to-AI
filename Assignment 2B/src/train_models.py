import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Embedding, Concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from datetime import timedelta

# === CONFIG ===
CSV_PATH = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"
MODEL_DIR = "../models"
RESULT_PATH = "../results/model_evaluation.csv"
SEQ_LENGTH = 24

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))
    v_cols = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

    df_long = df.melt(id_vars=["SCATS Number", "Date"], value_vars=v_cols, var_name="Bin", value_name="volume")
    df_long = df_long.sort_values(["SCATS Number", "Date", "Bin"]).reset_index(drop=True)
    df_long["hour"] = df_long["Bin"].apply(lambda x: int(x[1:]) // 4)
    df_long["minute_offset"] = df_long["Bin"].apply(lambda x: int(x[1:]) * 15)
    df_long["timestamp"] = pd.to_datetime(df_long["Date"]) + pd.to_timedelta(df_long["minute_offset"], unit="m")

    volume_scaler = MinMaxScaler()
    df_long["volume_scaled"] = volume_scaler.fit_transform(df_long["volume"].values.reshape(-1, 1))

    scats_encoder = LabelEncoder()
    df_long["scats_encoded"] = scats_encoder.fit_transform(df_long["SCATS Number"])

    sequences, scats_ids, targets, timestamps = [], [], [], []

    for scats_id, group in df_long.groupby("SCATS Number"):
        volumes = group["volume_scaled"].values
        ts = group["timestamp"].values
        scats_index = scats_encoder.transform([scats_id])[0]
        for i in range(len(volumes) - SEQ_LENGTH - 1):
            sequences.append(volumes[i:i + SEQ_LENGTH])
            targets.append(volumes[i + SEQ_LENGTH])
            scats_ids.append(scats_index)
            timestamps.append(ts[i + SEQ_LENGTH])

    X_seq = np.array(sequences)
    X_scats = np.array(scats_ids)
    y = np.array(targets)
    timestamps = np.array(timestamps)

    split_idx = int(len(X_seq) * 0.8)
    return (X_seq[:split_idx], X_scats[:split_idx], y[:split_idx], timestamps[:split_idx],
            X_seq[split_idx:], X_scats[split_idx:], y[split_idx:], timestamps[split_idx:],
            volume_scaler, scats_encoder)

def build_lstm_model(num_scats):
    seq_input = Input(shape=(SEQ_LENGTH, 1))
    scats_input = Input(shape=(1,))
    embed = Embedding(input_dim=num_scats, output_dim=4)(scats_input)
    embed_flat = Dense(1)(embed)
    lstm_out = LSTM(64)(seq_input)
    merged = Concatenate()([lstm_out, embed_flat[:, 0]])
    out = Dense(1)(merged)
    return Model(inputs=[seq_input, scats_input], outputs=out)

def build_gru_model(num_scats):
    seq_input = Input(shape=(SEQ_LENGTH, 1))
    scats_input = Input(shape=(1,))
    embed = Embedding(input_dim=num_scats, output_dim=4)(scats_input)
    embed_flat = Dense(1)(embed)
    gru_out = GRU(64)(seq_input)
    merged = Concatenate()([gru_out, embed_flat[:, 0]])
    out = Dense(1)(merged)
    return Model(inputs=[seq_input, scats_input], outputs=out)

def build_tcn_model(num_scats):
    seq_input = Input(shape=(SEQ_LENGTH, 1))
    scats_input = Input(shape=(1,))
    embed = Embedding(input_dim=num_scats, output_dim=4)(scats_input)
    embed_flat = Dense(1)(embed)
    conv_out = Conv1D(64, kernel_size=3, padding="causal", activation="relu")(seq_input)
    flat = Flatten()(conv_out)
    merged = Concatenate()([flat, embed_flat[:, 0]])
    out = Dense(1)(merged)
    return Model(inputs=[seq_input, scats_input], outputs=out)

def train_and_evaluate(model_name, build_fn, X_seq_train, X_scats_train, y_train,
                       X_seq_test, X_scats_test, y_test, timestamps_test,
                       scaler, scats_encoder):
    model = build_fn(len(scats_encoder.classes_))
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit([X_seq_train[..., np.newaxis], X_scats_train], y_train,
              epochs=50, batch_size=32, validation_split=0.1, callbacks=[es], verbose=1)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{model_name}_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/{model_name}_scaler.pkl")
    joblib.dump(scats_encoder, f"{MODEL_DIR}/{model_name}_scats_encoder.pkl")

    y_pred = model.predict([X_seq_test[..., np.newaxis], X_scats_test])
    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    # === Save true vs predicted flow for plotting
    flow_df = pd.DataFrame({
        "timestamp": timestamps_test,
        "true": y_true_inv.flatten(),
        "predicted": y_pred_inv.flatten()
    })
    flow_df.to_csv(f"../results/flow_{model_name.lower()}.csv", index=False)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)

    print(f"📊 {model_name.upper()} Results — MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return {
        "model": model_name.upper(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru", "tcn", "all"], default="all")
    args = parser.parse_args()

    (X_seq_train, X_scats_train, y_train, ts_train,
     X_seq_test, X_scats_test, y_test, ts_test,
     scaler, scats_encoder) = load_data()

    results = []
    if args.model in ["lstm", "all"]:
        results.append(train_and_evaluate("lstm", build_lstm_model,
                                          X_seq_train, X_scats_train, y_train,
                                          X_seq_test, X_scats_test, y_test, ts_test,
                                          scaler, scats_encoder))
    if args.model in ["gru", "all"]:
        results.append(train_and_evaluate("gru", build_gru_model,
                                          X_seq_train, X_scats_train, y_train,
                                          X_seq_test, X_scats_test, y_test, ts_test,
                                          scaler, scats_encoder))
    if args.model in ["tcn", "all"]:
        results.append(train_and_evaluate("tcn", build_tcn_model,
                                          X_seq_train, X_scats_train, y_train,
                                          X_seq_test, X_scats_test, y_test, ts_test,
                                          scaler, scats_encoder))

    os.makedirs("../results", exist_ok=True)
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
