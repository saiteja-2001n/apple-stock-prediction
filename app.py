import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# --- Min and Max values from training dataset ---
MIN_VALUES = {
    'open': 0.0388,
    'high': 0.0388,
    'low': 0.0384,
    'close': 0.0384,
    'volume': 1001504
}
MAX_VALUES = {
    'open': 181.8779,
    'high': 182.1866,
    'low': 178.3824,
    'close': 181.2605,
    'volume': 2147483647
}

def minmax_scale(value, col):
    return (value - MIN_VALUES[col]) / (MAX_VALUES[col] - MIN_VALUES[col])

# --- Load the trained model ---
model_path = "knn_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please place 'knn_model.pkl' in this folder.")
    st.stop()

with open(model_path, "rb") as file:
    model = pickle.load(file)

# --- Page Config ---
st.set_page_config(page_title="📈 Apple Stock Profit Predictor", layout="centered")

st.markdown("""
    ## 📘 Project Explanation
    This project predicts whether you'll make a **Profit** or **Loss** based on Apple's stock data using a **KNN classification model**.

    **Input Features:**
    - Open, High, Low, Close Prices 📊
    - Volume Traded 🔁

    **Output:**
    - 🔮 Prediction: Profit (1) or Loss (0)
    - 📌 Confidence Score (Probability)

    **ML Model Used:**  
    - 🧠 K-Nearest Neighbors (KNN)

    **Scaling Used:**
    - ✨ Manual MinMaxScaler
    """)

# --- Main UI ---
st.title("📊 Apple Stock Profit or Loss Prediction")
st.markdown("Use the sliders below to enter stock data. The app will predict whether you'll make a **Profit (1)** or **Loss (0)**.")

# Input sliders
open_price = st.slider("Open Price", MIN_VALUES['open'], MAX_VALUES['open'], step=0.1, key="open_price")
high_price = st.slider("High Price", MIN_VALUES['high'], MAX_VALUES['high'], step=0.1, key="high_price")
low_price = st.slider("Low Price", MIN_VALUES['low'], MAX_VALUES['low'], step=0.1, key="low_price")
close_price = st.slider("Close Price", MIN_VALUES['close'], MAX_VALUES['close'], step=0.1, key="close_price")
volume = st.slider("Volume", MIN_VALUES['volume'], MAX_VALUES['volume'], step=1000000, key="volume")


# --- Prediction Logic ---
if st.button("🔮 Predict"):
    # Apply manual MinMax Scaling
    scaled_input = np.array([[
        minmax_scale(open_price, 'open'),
        minmax_scale(high_price, 'high'),
        minmax_scale(low_price, 'low'),
        minmax_scale(close_price, 'close'),
        minmax_scale(volume, 'volume')
    ]])

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    st.session_state['profit_prob'] = round(probabilities[1] * 100, 2)
    st.session_state['loss_prob'] = round(probabilities[0] * 100, 2)

    # Output
    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Prediction: **Profit** 📈\n\n🟢 Confidence: {st.session_state['profit_prob']}%") 
        st.balloons()
    else:
        st.error(f"❌ Prediction: **Loss / No Profit** 📉\n\n🔴 Confidence: {st.session_state['loss_prob']}%")

# --- Probability Chart ---
if 'profit_prob' in st.session_state and 'loss_prob' in st.session_state:
    st.markdown('### 📊 Prediction Confidence Chart')
    prob_df = pd.DataFrame({
        'Outcome': ['Loss', 'Profit'],
        'Confidence (%)': [st.session_state['loss_prob'], st.session_state['profit_prob']]
    }).set_index('Outcome')

    st.bar_chart(prob_df)