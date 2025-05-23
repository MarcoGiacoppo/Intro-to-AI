import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model #type:ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Embedding, Concatenate, Lambda #type:ignore
from tensorflow.keras.callbacks import EarlyStopping #type:ignore
import tensorflow.keras.backend as K #type:ignore
import joblib
from datetime import timedelta
from keras.saving import register_keras_serializable #type:ignore
import tensorflow.keras.backend as K #type:ignore
from tcn import TCN


@register_keras_serializable(package="Custom")
class RegisteredTCN(TCN):
    pass

@register_keras_serializable(package="Custom")
def squeeze_dim1(x):
    return K.squeeze(x, axis=1)

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

def build_tcn_model(num_scats, seq_length):
    # Input 1: Traffic sequence 
    seq_input = Input(shape=(seq_length, 1))

    # Input 2: SCATS site ID
    scats_input = Input(shape=(1,)) 

    # Embed the SCATS site ID
    embed = Embedding(input_dim=num_scats, output_dim=4)(scats_input) 
    embed_flat = Dense(1)(embed) 
    embed_squeezed = Lambda(squeeze_dim1, name="squeeze_dim1")(embed_flat)  

    # Temporal Convolutional Network for sequence modeling
    tcn_out = RegisteredTCN(
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        padding="causal",
        return_sequences=False
    )(seq_input)  # shape: (batch, features)

    # Merge TCN output with SCATS site embedding
    merged = Concatenate()([tcn_out, embed_squeezed])

    # Final regression output: estimated travel time
    out = Dense(1)(merged)

    return Model(inputs=[seq_input, scats_input], outputs=out)

def train_and_evaluate(model_name, build_fn, X_seq_train, X_scats_train, y_train,
                       X_seq_test, X_scats_test, y_test, timestamps_test,
                       scaler, scats_encoder):
    import time

    model = build_fn(len(scats_encoder.classes_))
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # === Train
    start = time.time()
    history = model.fit(
        [X_seq_train[..., np.newaxis], X_scats_train],
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )
    end = time.time()

    total_time = end - start
    epochs_run = len(history.history["loss"])
    time_per_epoch = total_time / epochs_run if epochs_run else 0
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    # === Save loss per epoch
    history_df = pd.DataFrame({
        "epoch": range(1, epochs_run + 1),
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    })
    os.makedirs("../results", exist_ok=True)
    history_df.to_csv(f"../results/loss_curve_{model_name.lower()}.csv", index=False)

    # === Save model + scalers
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{model_name}_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/{model_name}_scaler.pkl")
    joblib.dump(scats_encoder, f"{MODEL_DIR}/{model_name}_scats_encoder.pkl")

    # === Predict
    y_pred = model.predict([X_seq_test[..., np.newaxis], X_scats_test])
    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    flow_df = pd.DataFrame({
        "timestamp": timestamps_test,
        "true": y_true_inv.flatten(),
        "predicted": y_pred_inv.flatten()
    }).sort_values("timestamp")
    flow_df.to_csv(f"../results/flow_{model_name.lower()}.csv", index=False)

    # === Evaluation
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)
    num_layers = len([l for l in model.layers if not l.__class__.__name__.startswith("Input")])

    print(f"📊 {model_name.upper()} Results — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return {
        "model": model_name.upper(),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
        "TrainingTimePerEpoch": round(time_per_epoch, 2),
        "NumLayers": num_layers,
        "FinalLoss": round(final_loss, 6),
        "FinalValLoss": round(final_val_loss, 6),
        "EpochsRun": epochs_run,
        "TotalTrainingTime": round(total_time, 2)
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
        results.append(train_and_evaluate("tcn", lambda n: build_tcn_model(n, SEQ_LENGTH),
                                          X_seq_train, X_scats_train, y_train,
                                          X_seq_test, X_scats_test, y_test, ts_test,
                                          scaler, scats_encoder))

    os.makedirs("../results", exist_ok=True)
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
