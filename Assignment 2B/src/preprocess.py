import pandas as pd
import os

INPUT_XLS = "../data/raw/Scats Data October 2006.xls"
OUTPUT_CSV = "../data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv"

def extract_all_required_columns():
    print("📥 Reading Excel file...")
    df = pd.read_excel(INPUT_XLS, sheet_name="Data", header=1)

    # Standardize column names
    df.columns = [str(col).strip() for col in df.columns]

    # Identify all V00–V95 columns
    v_cols = [col for col in df.columns if col.startswith("V") and col[1:].isdigit()]

    # Full list of required non-V columns (as per your screenshot)
    meta_cols = [
        "SCATS Number", "Location", "CD_MELWAY",
        "NB_LATITUDE", "NB_LONGITUDE",
        "HF VicRoads Internal", "VR Internal Stat", "VR Internal Loc",
        "NB_TYPE_SURVEY", "Date"
    ]

    # Final ordered column list
    all_cols = meta_cols + v_cols

    print(f"✅ Extracting {len(all_cols)} columns...")
    df_clean = df[all_cols]
    df_clean["SCATS Number"] = df_clean["SCATS Number"].apply(lambda x: str(x).zfill(4))
    df_clean["Date"] = pd.to_datetime(df_clean["Date"]).dt.date  # Strip time part

    print("💾 Saving full raw format with metadata + V columns...")
    os.makedirs("../data/processed", exist_ok=True)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_all_required_columns()