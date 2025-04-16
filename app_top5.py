
import streamlit as st
import numpy as np
import joblib

# Load top 5 feature pipeline components
model = joblib.load("ridge_classifier_model_top5.pkl")
scaler = joblib.load("scaler_top5.pkl")
selector = joblib.load("feature_selector_top5.pkl")

# Define input fields
st.title("Supply Chain Risk Classifier (Top 5 Features)")

st.write("Enter the following logistics parameters:")

features = [
    "warehouse_inventory_level",
    "order_fulfillment_status",
    "weather_condition_severity",
    "cargo_condition_status",
    "disruption_likelihood_score"
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
