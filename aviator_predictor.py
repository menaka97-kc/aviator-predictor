import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import random

st.set_page_config(page_title="Aviator Predictor AI", layout="centered")
st.title("✈️ Aviator Predictor AI (XGBoost Enhanced)")
st.markdown("Get accurate crash predictions using advanced AI.")

# Load XGBoost model with same structure
def generate_crash_data(n=5000):
    data = []
    for _ in range(n):
        prev = round(np.random.uniform(1.0, 10.0), 2)
        second = round(np.random.uniform(1.0, 10.0), 2)
        last_5 = [round(np.random.uniform(1.0, 10.0), 2) for _ in range(5)]
        avg_5 = round(np.mean(last_5), 2)
        std_5 = round(np.std(last_5), 2)
        crash_label = 1 if avg_5 < 2.0 or prev < 1.5 else 0
        data.append([prev, second, avg_5, std_5, crash_label])
    return pd.DataFrame(data, columns=["prev_crash", "second_last_crash", "avg_last_5", "volatility_score", "label"])

df = generate_crash_data()
X = df.drop("label", axis=1)
y = df["label"]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)

# Input UI
st.subheader("📊 Input Crash Values")
prev = st.slider("Previous Crash", 0.0, 10.0, 2.0)
second = st.slider("Second Last Crash", 0.0, 10.0, 2.0)
last_5 = [prev, second] + [round(np.random.uniform(1.0, 10.0), 2) for _ in range(3)]
avg_5 = round(np.mean(last_5), 2)
std_5 = round(np.std(last_5), 2)

st.write("Avg of Last 5:", avg_5, "| Volatility Score:", std_5)

# Predict
input_data = pd.DataFrame([[prev, second, avg_5, std_5]], columns=X.columns)
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0][1]

result_text = "🚨 Likely Crash!" if prediction == 1 else "✅ Safe Zone"
color = "red" if prediction == 1 else "green"
st.markdown(f"<h3 style='color:{color}'>{result_text}</h3>", unsafe_allow_html=True)
st.info(f"Confidence: {proba * 100:.2f}%")

# Animation
st.subheader("🎮 Crash Animation (Mock)")
with st.empty():
    for i in range(1, 21):
        time.sleep(0.1)
        st.metric("Plane Altitude", f"{i/2:.1f}x")
st.warning("🔥 Plane crashed at simulated 9.2x")
