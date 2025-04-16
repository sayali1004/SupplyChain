
import streamlit as st
import numpy as np
import joblib

# Load saved models
model = joblib.load("ridge_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("feature_selector.pkl")

# Define input fields
st.title("Supply Chain Risk Classifier")

st.write("Enter the following logistics parameters:")

features = [
    "eta_variation_hours",
    "traffic_congestion_level",
    "warehouse_inventory_level",
    "order_fulfillment_status",
    "weather_condition_severity",
    "port_congestion_level",
    "lead_time_days",
    "iot_temperature",
    "cargo_condition_status",
    "route_risk_level",
    "driver_behavior_score",
    "fatigue_monitoring_score",
    "disruption_likelihood_score",
    "delay_probability",
    "delivery_time_deviation"
]

user_input = []
for feature in features:
    value = st.number_input(f"{feature.replace('_', ' ').capitalize()}", format="%.4f")
    user_input.append(value)

if st.button("Predict Risk"):
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    prediction = model.predict(X_selected)[0]
    
    risk_labels = {0: "High Risk", 1: "Low Risk", 2: "Moderate Risk"}
    st.success(f"Predicted Risk Classification: **{risk_labels[prediction]}**")
