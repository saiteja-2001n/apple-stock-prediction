# apple-stock-prediction

A machine learning web application built with **Streamlit** that predicts whether a trade on Apple (AAPL) stock will result in a **Profit** or **Loss** based on specific market indicators.

## 🚀 Live Demo
https://huggingface.co/spaces/saiteja2001/Apple_Stock_Prediction

## ✨ Features
* **Real-time Prediction:** Input stock parameters (Open, High, Low, Close, Volume) to get an instant prediction.
* **Confidence Scoring:** Shows the probability percentage for both Profit and Loss outcomes.
* **ML Powered:** Uses a trained **K-Nearest Neighbors (KNN)** model for classification.
* **User-Friendly UI:** Simple sliders for easy data input.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** Scikit-learn (K-Neighbors Classifier)
* **Data Processing:** Pandas, NumPy
* **Model Persistence:** Pickle

## 📂 Project Structure

* `app.py`: The main Streamlit application script.
* `knn_model.pkl`: The trained machine learning model.
* `requirements.txt`: List of required Python libraries.
* `README.md`: Project documentation.

## 📊 How it Works

The model takes 5 inputs: **Open, High, Low, Close Prices, and Volume**. It scales these values using a pre-defined Min-Max scaling logic based on the original training dataset and then feeds them into the KNN model to determine the likely outcome.



