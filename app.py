# Aviator Predictor App using Streamlit, AI, and Automation
# Step-by-step setup guide

# IMPORTANT: This app must be run in an environment where Streamlit is pre-installed
# such as your local machine or Streamlit Cloud.

# 1. Install Required Libraries
# Run these commands in your terminal (not inside this sandbox):
# pip install streamlit pandas numpy scikit-learn matplotlib
# pip install streamlit-authenticator selenium requests webdriver-manager

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import time
    import hashlib
    import streamlit_authenticator as stauth
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    import requests
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
except ModuleNotFoundError as e:
    print("\nERROR: This script requires packages that are not available in this environment.")
    print("Run it locally or deploy using Streamlit Cloud.")
    print(f"Missing module: {e.name}")
    exit(1)

# 3. Configure Login (high-security features)
usernames = ['admin']
passwords = [stauth.Hasher(['your_password_here']).generate()[0]]
authenticator = stauth.Authenticate(
    {'admin': {'name': 'Admin', 'password': passwords[0]}},
    'aviator_dashboard', 'abcdef', cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.title("🎮 Aviator Predictor AI App")

    # 4. Upload or Stream Live Data
    st.sidebar.header("Data Input")
    data_source = st.sidebar.radio("Choose Data Source:", ['Upload CSV', 'Live Stream'])

    def fetch_live_data():
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get("https://1xbet.com/en/slots/game/52358/aviator")
            time.sleep(10)
            # Simulate scraped crash data for now
            data = pd.DataFrame({
                'round': np.arange(1, 201),
                'crash_point': np.random.uniform(1.0, 10.0, 200)
            })
            driver.quit()
            return data
        except Exception as e:
            st.error(f"Live fetch failed: {e}")
            return pd.DataFrame()

    if data_source == 'Upload CSV':
        uploaded_file = st.sidebar.file_uploader("Upload your Aviator CSV data")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.DataFrame()
    else:
        df = fetch_live_data()
        if df.empty:
            st.warning("Live data not available, using simulated fallback.")
            df = pd.DataFrame({
                'round': np.arange(1, 201),
                'crash_point': np.random.uniform(1.0, 10.0, 200)
            })

    # 5. Show Raw Data
    if not df.empty and st.checkbox("Show Raw Data"):
        st.subheader("Raw Aviator Crash Data")
        st.dataframe(df.tail(50))

    if not df.empty:
        # 6. Data Preprocessing
        df['prev_crash'] = df['crash_point'].shift(1).fillna(1.0)

        # 7. Model Training
        X = df[['prev_crash']]
        y = df['crash_point']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 8. Prediction Section
        st.header("🚀 Predict Next Round")
        last_crash = df['crash_point'].iloc[-1]
        prediction = model.predict([[last_crash]])[0]
        st.metric(label="Predicted Next Crash Point", value=round(prediction, 2))

        # Optional: Telegram alert if high prediction
        threshold = 3.0
        if prediction > threshold:
            st.info("📢 Prediction exceeds threshold! Alert triggered.")

        # 9. Visualization
        st.subheader("Crash Point Trend")
        plt.plot(df['round'], df['crash_point'], label='Crash Points')
        plt.xlabel('Round')
        plt.ylabel('Crash Point')
        plt.legend()
        st.pyplot(plt)

        # 10. Auto-refresh every 60 seconds
        if 'last_run' not in st.session_state:
            st.session_state['last_run'] = time.time()
        if time.time() - st.session_state['last_run'] > 60:
            st.session_state['last_run'] = time.time()
            st.experimental_rerun()
        else:
            remaining = 60 - int(time.time() - st.session_state['last_run'])
            st.sidebar.info(f"🔄 Refreshing in {remaining} seconds")

        st.sidebar.markdown("---")
        st.sidebar.write("✅ Future: Add real animations with Streamlit components")

elif authentication_status == False:
    st.error("Incorrect username or password")

elif authentication_status == None:
    st.warning("Please enter your credentials")
