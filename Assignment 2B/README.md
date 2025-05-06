How to use the program:

1. Install Required Libraries
   Make sure you have Python 3 installed. Install all the required packages by running:
   pip install pandas numpy scikit-learn matplotlib tensorflow

2. Prepare the Dataset
   Place the preprocessed traffic dataset called "Oct_2006_Boorondara_Traffic_Flow_Data.csv" inside the folder:
   data/processed/

if not there, just run preprocess.py

3. Train the LSTM Model
   Run the training script by typing:
   python src/train_lstm.py
   This script will:

Train a model on traffic volumes for SCATS site 0970

Save the trained model to: models/lstm_model.h5

Save the evaluation results to: results/model_evaluation.csv

4. Find the Best Route Between Two SCATS Sites
   Run the route finder script:
   python src/route_finder.py
   The program will prompt you to enter:

An origin SCATS number (e.g., 2000)

A destination SCATS number (e.g., 3002)
It will then:

Predict traffic flow at each site

Convert traffic volume into estimated travel time

Use A\* to find the best path

Show the full route with step-by-step site IDs, time taken, and connected roads

5. Visualize the Route (Optional)
   If you want to see the route drawn on a map using latitude and longitude:

Open the script "visualize_route.py"

Run:
python src/visualize_route.py
pick origin and destination
