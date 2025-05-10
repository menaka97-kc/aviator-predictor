import random
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
API_URL = "https://example.com/api/aviator-crash-history"  # Replace with real API

# Simulated data for training
def generate_synthetic_data(n=10000):
    return pd.DataFrame({
        'previous_crash': [round(random.uniform(1.0, 10.0), 2) for _ in range(n)],
        'second_last_crash': [round(random.uniform(1.0, 10.0), 2) for _ in range(n)],
        'avg_last_5': [round(random.uniform(1.0, 10.0), 2) for _ in range(n)],
        'target': [1 if random.random() > 0.5 else 0 for _ in range(n)]
    })

# Fetch live data (mocked)
def fetch_live_data():
    try:
        # response = requests.get(API_URL)
        # crash_data = response.json()
        crash_data = [round(random.uniform(1.0, 10.0), 2) for _ in range(10)]
        return crash_data[0], crash_data[1], round(sum(crash_data[:5]) / 5, 2)
    except:
        return 2.0, 2.0, 2.0

# Train model
st.info("Training model...")
data = generate_synthetic_data()
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model trained with {accuracy*100:.2f}% accuracy")

# Prediction logic
def predict_next_crash(prev, sec_last, avg5):
    input_df = pd.DataFrame([[prev, sec_last, avg5]], columns=X.columns)
    prediction = model.predict(input_df)
    return 'High Crash Likely' if prediction[0] == 1 else 'Low Crash Likely'

# Streamlit UI
st.title("Aviator Predictor AI")

option = st.radio("Choose Input Mode", ["Manual Input", "Live Data"])

if option == "Manual Input":
    prev = st.number_input("Previous Crash", min_value=0.0, max_value=100.0, value=2.0)
    sec = st.number_input("Second Last Crash", min_value=0.0, max_value=100.0, value=2.0)
    avg5 = st.number_input("Average of Last 5", min_value=0.0, max_value=100.0, value=2.0)
else:
    prev, sec, avg5 = fetch_live_data()
    st.write(f"Fetched Values - Previous: {prev}, Second Last: {sec}, Avg 5: {avg5}")

if st.button("Predict"):
    result = predict_next_crash(prev, sec, avg5)
    st.header(f"Prediction: {result}")
