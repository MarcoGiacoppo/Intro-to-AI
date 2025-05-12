# Traffic-Based Route Guidance System (TBRGS)

This project implements a machine learning-enhanced route guidance system for the Boroondara area. It includes search algorithms, ML traffic prediction models, visualizations, and a GUI for user interaction.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                         # Original SCATS datasets
│   ├── processed/                   # Cleaned and structured dataset
│   ├── graph/                       # Generated adjacency and metadata files
├── models/                          # Trained ML models (LSTM, GRU, TCN)
├── results/                         # Evaluation results and predicted traffic flow CSVs
├── images/                          # Plots and visualizations for the report
├── src/                             # All source code files
│   ├── train_models.py              # Train and evaluate ML models
│   ├── display_route_map.py         # Maps route on streamlit app
│   ├── gui_streamlit.py            # Interactive GUI for user input and route visualization
│   ├── generate_adjacency.py       # Build graph from SCATS site links
│   ├── generate_sites_metadata.py  # Create coordinates and metadata
│   ├── preprocess.py               # Prepares the dataset for training
│   └── search_algorithms.py        # DFS, BFS, UCS, A*, GBFS algorithms
├── visuals/                             # Source Code for visuals
│   ├── plot_error_heatmap.py       
│   ├── plot_metrics_bar.py          
│   ├── plot_predicted_vs_true_split.py
│   ├── plot_time_series_comparison.py     
```

---

## 🛠 Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**

   ```bash
   python3 src/generate_adjacency.py
   python3 src/generate_sites_metadata.py
   ```

3. **Preprocess traffic data**

   ```bash
   python3 src/preprocess.py
   ```

4. **Train all ML models (LSTM, GRU, TCN)**
   ```bash
   python3 src/train_models.py --model all
   ```
   Trained models and scalers will be saved to `/models`. Flow predictions are saved to `/results`.

---

## 🧠 Model Evaluation and Visualization

### Generate Evaluation Table

No action needed — metrics like MAE, RMSE, R², MAPE are automatically saved to:

```
/results/model_evaluation.csv
```

### Plot Time Series

```bash
python3 src/plot_time_series_comparison.py
```

Generates:

```
/images/flow_time_series_comparison.png
```

---

## 🧭 Route Finding

### Terminal (CLI) Mode

```bash
python3 src/route_finder.py
```

Allows search via CLI with algorithm and node input.

### GUI Mode (Preferred)

```bash
python3 src/gui.py
```

- Input origin and destination SCATS site numbers
- Select ML model and search algorithm
- View best path and travel time estimate
- Route is plotted on a 2D map

---

## 📊 How Travel Time is Predicted

1. Trained ML models predict traffic volume at a given SCATS site
2. Volume is converted to speed via a parabolic formula
3. Travel time is computed as `60 / speed`

---

## ✅ Evaluation Metrics

Stored in `/results/model_evaluation.csv` with:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

You can compare them using `plot_metric_comparison.py`

---
