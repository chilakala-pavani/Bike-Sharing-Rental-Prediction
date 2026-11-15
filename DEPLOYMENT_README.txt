# Bike-Sharing Rental Demand Prediction — Deployment Package

## Files
- `app.py` — Streamlit web application
- `bike_demand_model.joblib` — trained preprocessing + Random Forest model artifact
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Deploy using Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `bike_demand_model.joblib`, and `requirements.txt`.
3. In Streamlit Community Cloud, create an app and select the repository and `app.py`.
4. Deploy.

The model was trained from the supplied `Dataset (1).csv` using a chronological 80/20 split.
The app uses the same engineered feature columns and preprocessing pipeline as the training workflow.
