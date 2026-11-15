import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="centered")

@st.cache_resource
def load_artifact():
    return joblib.load("bike_demand_model.joblib")

artifact = load_artifact()
model = artifact["model"]

def season_from_month(month):
    # Matches the season labels used in the supplied training dataset.
    if month in [1, 2, 3]:
        return "spring"
    if month in [4, 5, 6]:
        return "summer"
    if month in [7, 8, 9]:
        return "fall"
    return "winter"

st.title("🚲 Bike-Sharing Rental Demand Prediction")
st.write("Enter the date, time and weather conditions to estimate hourly bike rentals.")

with st.form("prediction_form"):
    chosen_date = st.date_input(
        "Date",
        value=date(2012, 6, 15),
        min_value=date(2011, 1, 1),
        max_value=date(2012, 12, 31)
    )
    hour = st.slider("Hour of the day", 0, 23, 8)
    season = st.selectbox("Season", ["spring", "summer", "fall", "winter"],
                          index=["spring", "summer", "fall", "winter"].index(season_from_month(chosen_date.month)))
    weather = st.selectbox("Weather situation", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])
    workingday = st.selectbox("Is it a working day?", ["Working Day", "No work"])
    temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.50, 0.01)
    atemp = st.slider("Feels-like temperature (normalized)", 0.0, 1.0, 0.48, 0.01)
    hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.60, 0.01)
    windspeed = st.slider("Wind speed (normalized)", 0.0, 1.0, 0.20, 0.01)
    submitted = st.form_submit_button("Predict rental demand")

if submitted:
    # UCI hourly dataset convention: yr=0 for 2011 and yr=1 for 2012;
    # weekday is Sunday=0 through Saturday=6.
    weekday_sunday_zero = (chosen_date.weekday() + 1) % 7
    row = pd.DataFrame([{
        "season": season,
        "yr": chosen_date.year - 2011,
        "mnth": chosen_date.month,
        "hr": hour,
        "holiday": holiday,
        "weekday": weekday_sunday_zero,
        "workingday": workingday,
        "weathersit": weather,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "dteday": pd.Timestamp(chosen_date)
    }])

    x = row.copy()
    x["dteday"] = pd.to_datetime(x["dteday"])
    x["year"] = x["dteday"].dt.year
    x["month_from_date"] = x["dteday"].dt.month
    x["day"] = x["dteday"].dt.day
    x["dayofyear"] = x["dteday"].dt.dayofyear
    x["dayofweek_from_date"] = x["dteday"].dt.dayofweek
    x["is_weekend"] = (x["dayofweek_from_date"] >= 5).astype(int)
    x["hour_sin"] = np.sin(2 * np.pi * x["hr"].astype(float) / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hr"].astype(float) / 24)
    x["month_sin"] = np.sin(2 * np.pi * x["month_from_date"].astype(float) / 12)
    x["month_cos"] = np.cos(2 * np.pi * x["month_from_date"].astype(float) / 12)
    x = x.drop(columns=["dteday"])
    x = x[artifact["feature_columns"]]

    prediction = max(0, float(model.predict(x)[0]))
    st.success(f"Estimated rentals: {prediction:,.0f} bikes")
    st.caption(f"Model: {artifact['model_name']}")

st.divider()
st.caption("Academic project demonstration. Predictions are estimates and depend on the entered conditions.")
