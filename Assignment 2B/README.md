# Traffic-Based Route Guidance System (TBRGS)

This project implements a machine learning-enhanced route guidance system for the Boroondara area. It includes classic search algorithms, traffic prediction models, insightful visualizations, and an interactive GUI.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                         # Original SCATS datasets
│   ├── processed/                   # Cleaned and structured dataset
│   ├── graph/                       # Generated adjacency and metadata files
├── models/                          # Trained ML models (LSTM, GRU, TCN)
├── results/                         # Evaluation results, flow predictions, training loss CSVs
├── images/                          # Plots and visualizations for the report
├── src/                             # All source code files
│   ├── train_models.py              # Train and evaluate ML models
│   ├── display_route_map.py         # Maps route on streamlit app
│   ├── gui_streamlit.py             # Interactive GUI for user input and route visualization
│   ├── generate_adjacency.py        # Build graph from SCATS site links
│   ├── generate_sites_metadata.py   # Create coordinates and metadata
│   ├── preprocess.py                # Prepares the dataset for training
│   └── search_algorithms.py         # DFS, BFS, UCS, A*, GBFS algorithms
├── visuals/                         # Visualization scripts
│   ├── plot_error_heatmap.py
│   ├── plot_error_histogram.py
│   ├── plot_error_over_time.py
│   └── plot_loss_curve.py
│   ├── plot_metrics_bar.py
│   ├── plot_predicted_vs_true_split.py
│   └── plot_prediction_distribution.py
│   ├── plot_time_series_comparison.py
```

---

## 🚀 Setup Instructions

---

### 🖱️ One-Click Setup & Launch (Windows Only)

If you're on Windows, you can skip all manual setup steps by using the provided batch script:

```bash
setup_and_run.bat
```

### 🔧 1. Python Version
This project requires **Python 3.11** for compatibility with TensorFlow 2.19.  
TensorFlow does **not yet support Python 3.12 or newer**.

📥 [Download Python 3.11.5](https://www.python.org/downloads/release/python-3115/) if needed.

---

### 📦 2. Virtual Environment Setup

#### ✅ On **Windows**:

From the project root (`Assignment 2B` folder), run:

```bash
"C:\Program Files\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

💡 If you get a PowerShell script error, run:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### ✅ On **Mac/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🛠️ 3. Build Pipeline

Navigate to the `src/` folder:

```bash
cd src/
```

Run the following scripts **in order**:

```bash
python3 generate_sites_metadata.py
python3 generate_adjacency.py
python3 preprocess.py
```

This will generate:
- `data/graph/sites_metadata.json`
- `data/graph/adjacency_from_summary.json`
- `data/processed/Oct_2006_Boorondara_Traffic_Flow_Data.csv`

---

### 4. Train all ML models (LSTM, GRU, TCN)

```bash
python3 train_models.py --model all
```

#### ✅ Outputs

- Trained model files → `../models/`
- Predicted flows → `../results/flow_*.csv`
- Per-epoch loss → `../results/loss_curve_*.csv`
- Evaluation metrics → `../results/model_evaluation.csv`

---

## 🧠 Evaluation and Visualization

> 📌 For all visualization scripts, navigate to the `visuals/` directory first:

```bash
cd ../visuals/
```

### 📉 Metrics Table

Automatically saved to:

```bash
../results/model_evaluation.csv
```

### 📊 Visualizations

Run the following scripts to generate plots:

```bash
python plot_time_series_comparison.py
python plot_error_heatmap.py
python plot_predicted_vs_true_split.py
python plot_metrics_bar.py
python plot_loss_curve.py
```

Outputs go to the `../images/` directory.

---

## 💾 Launch the GUI

> From the `src/` directory:

```bash
streamlit run gui_streamlit.py
```

### Features:

- Select origin & destination SCATS site
- Choose ML model and search algorithm
- View estimated travel time and route steps
- Visualize the route on an interactive map

---

## 🧮 Travel Time Prediction Process

1. ML model predicts traffic **volume** at a SCATS site
2. Volume is converted to **speed** using a parabolic formula
3. Travel time = `distance / speed` (converted to minutes)

---

## 📌 Notes for Cross-Platform Use

- Use `python3` on Mac/Linux, and `python` or `py -3.11` on Windows depending on your install.
- Make sure your terminal is in the root project folder when running commands.
- If using OneDrive or synced folders, avoid filename issues by sticking to ASCII filenames.

---

## ✅ Evaluation Metrics Explained

All metrics are saved in `../results/model_evaluation.csv`:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination
- **MAPE** – Mean Absolute Percentage Error
- **Final Loss / Val Loss**
- **Training Time per Epoch**

Use `plot_metrics_bar.py` and `plot_loss_curves.py` for visual comparison.
