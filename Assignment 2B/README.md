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
│   ├── plot_metrics_bar.py
│   ├── plot_predicted_vs_true_split.py
│   ├── plot_time_series_comparison.py
│   └── plot_loss_curves.py
```

---

## 🛠️ Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate metadata and adjacency**

   ```bash
   python3 src/generate_sites_metadata.py
   python3 src/generate_adjacency.py
   ```

3. **Preprocess traffic data**

   ```bash
   python3 src/preprocess.py
   ```

4. **Train all ML models (LSTM, GRU, TCN)**

   ```bash
   python3 src/train_models.py --model all
   ```

   Outputs:

   - Trained model files → `/models`
   - Predicted flows → `/results/flow_*.csv`
   - Per-epoch loss → `/results/loss_curve_*.csv`
   - Evaluation metrics → `/results/model_evaluation.csv`

---

## 🧠 Evaluation and Visualization

### 📉 Metrics Table

No extra steps — automatically saved to:

```
/results/model_evaluation.csv
```

### 📊 Visualizations

Run each script to generate the following:

- **Time Series Comparison**

  ```bash
  python3 visuals/plot_time_series_comparison.py
  ```

  → `/images/flow_time_series_comparison_avg.png`

- **Error Heatmap**

  ```bash
  python3 visuals/plot_error_heatmap.py
  ```

  → `/images/error_heatmap_lstm.png`
  → `/images/error_heatmap_gru.png`
  → `/images/error_heatmap_tcn.png`

- **Per-Site Predictions**

  ```bash
  python3 visuals/plot_predicted_vs_true_split.py
  ```

  → `/images/predicted_vs_true_split.png`, etc.

- **Model Metric Comparison**

  ```bash
  python3 visuals/plot_metrics_bar.py
  ```

  → `/images/metrics_comparison.png`

- **Loss Curve**

  ```bash
  python3 visuals/plot_loss_curves.py
  ```

  → `/images/loss_curves_all_models.png`

---

## 💾 GUI Mode

```bash
python3 src/gui_streamlit.py
```

Features:

- Select origin and destination SCATS site
- Choose ML model and search algorithm
- Displays best route and travel time
- Interactive route map with color-coded paths

---

## 🧮 How Travel Time Is Predicted

1. ML model predicts volume at a SCATS site
2. Volume → speed using a parabolic formula
3. Travel time = `distance / speed`

---

## ✅ Evaluation Metrics

Saved in `/results/model_evaluation.csv`:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination
- **MAPE** – Mean Absolute Percentage Error
- **Final Loss / Val Loss**
- **Training Time / Epoch**

You can compare models visually via the bar chart or loss curves.
