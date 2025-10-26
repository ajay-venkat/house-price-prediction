import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

# ------------------------------------------------
# Load model, scaler, and metadata
# ------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model('house_price_model.h5')
    scaler = joblib.load('scaler.gz')
    with open('model_metadata.json', 'r') as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_resources()
feature_names = meta['feature_names']

# ------------------------------------------------
# Streamlit App
# ------------------------------------------------
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction using Deep Learning")
st.markdown("Enter the house features below to predict its approximate price (in $1000s).")

# Function to collect user input
def user_input_features():
    inputs = {}
    for feat in feature_names:
        inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.3f")
    return pd.DataFrame([inputs])

input_df = user_input_features()

if st.button("Predict Price"):
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input).flatten()[0]
    st.success(f"🏡 Predicted Price: **${pred*1000:,.2f} USD**")
    st.caption(f"Model output: {pred:.2f} (in $1000s)")

st.divider()
st.write("**Tech Stack:** TensorFlow • Streamlit • Scikit-learn")
st.caption("This is a mini deep learning regression project using the Boston Housing dataset.")
