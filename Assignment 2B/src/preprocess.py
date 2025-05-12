import pandas as pd
import os

INPUT_XLS = "../data/raw/Scats Data October 2006.xls"
OUTPUT_CSV = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"

def preprocess_booroondara_data():
    print("📥 Reading Excel file...")
    df = pd.read_excel(INPUT_XLS, sheet_name="Data", header=1)
    df.columns = [str(col).strip() for col in df.columns]

    # Identify all V00–V95 columns
    v_cols = sorted([col for col in df.columns if col.startswith("V") and col[1:].isdigit()],
                    key=lambda x: int(x[1:]))

    # Columns to retain
    keep_cols = ["SCATS Number", "Location", "Date", "NB_LATITUDE", "NB_LONGITUDE"] + v_cols
    df = df[keep_cols].copy()
    df["SCATS Number"] = df["SCATS Number"].apply(lambda x: str(x).zfill(4))
    df["Date"] = pd.to_datetime(df["Date"]).dt.date  # Keep date only

    # Melt to long format: one row per 15-minute bin
    df_long = df.melt(
        id_vars=["SCATS Number", "Location", "Date", "NB_LATITUDE", "NB_LONGITUDE"],
        value_vars=v_cols,
        var_name="Bin",
        value_name="volume"
    )

    # Derive interval_id and time
    df_long["interval_id"] = df_long["Bin"].apply(lambda x: int(x[1:]))
    df_long["time"] = df_long["interval_id"].apply(lambda m: f"{(m * 15)//60:02}:{(m * 15)%60:02}")

    # Final reordering and renaming
    df_long = df_long.rename(columns={
        "SCATS Number": "site_id",
        "Location": "location",
        "NB_LATITUDE": "latitude",
        "NB_LONGITUDE": "longitude",
        "Date": "date"
    })[["site_id", "location", "date", "latitude", "longitude", "interval_id", "time", "volume"]]

    df_long = df_long.sort_values(["site_id", "location", "date", "time"]).reset_index(drop=True)

    print("💾 Saving streamlined file...")
    os.makedirs("../data/processed", exist_ok=True)
    df_long.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    preprocess_booroondara_data()
