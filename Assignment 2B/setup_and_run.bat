@echo off
REM === Full Setup: venv + Install + Preprocessing + Streamlit GUI ===

echo =============================================
echo 📦 Setting up virtual environment and running full pipeline
echo =============================================

REM Step 1: Check if venv exists
IF NOT EXIST "venv" (
    echo 🔧 Creating virtual environment using Python 3.11...
    "C:\Program Files\Python311\python.exe" -m venv venv
)

REM Step 2: Activate venv and install requirements
call venv\Scripts\activate.bat
echo 📦 Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

REM Step 3: Move to src folder
cd src

REM Step 4: Run preprocessing scripts
echo ⚙️  Running generate_sites_metadata.py...
python generate_sites_metadata.py

echo ⚙️  Running generate_adjacency.py...
python generate_adjacency.py

echo ⚙️  Running preprocess.py...
python preprocess.py

REM Step 5: Launch Streamlit GUI
echo 🚀 Launching Streamlit GUI...
streamlit run gui_streamlit.py

pause
