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

## 🛠️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Navigate to the `src/` directory

```bash
cd src/
```

### 3. Generate metadata and adjacency files

```bash
python3 generate_sites_metadata.py
python3 generate_adjacency.py
```

### 4. Preprocess traffic data

```bash
python3 preprocess.py
```

### 5. Train all ML models (LSTM, GRU, TCN)

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

Run each script below to generate and save images:

#### 🕒 Time Series Comparison

```bash
python3 plot_time_series_comparison.py
```

→ `../images/flow_time_series_comparison_avg.png`

#### 🔥 Error Heatmaps

```bash
python3 plot_error_heatmap.py
```

→ `../images/error_heatmap_lstm.png`, `error_heatmap_gru.png`, `error_heatmap_tcn.png`

#### 🧍 Per-Site Predictions

```bash
python3 plot_predicted_vs_true_split.py
```

→ `../images/predicted_vs_true_split.png`

#### 📊 Model Metric Comparison

```bash
python3 plot_metrics_bar.py
```

→ `../images/metrics_comparison.png`

#### 📉 Loss Curves

```bash
python3 plot_loss_curves.py
```

→ `../images/loss_curves_all_models.png`

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

## ✅ Evaluation Metrics Explained

All metrics are saved in `../results/model_evaluation.csv`:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination
- **MAPE** – Mean Absolute Percentage Error
- **Final Loss / Val Loss**
- **Training Time per Epoch**

Use `plot_metrics_bar.py` and `plot_loss_curves.py` for visual comparison.
