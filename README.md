# 🧠 Aviator Predictor App (AI + Streamlit)

This AI-powered Aviator Predictor app uses machine learning and real-time web scraping to predict the next crash point of the Aviator game. Built with Streamlit and deployed on the cloud.

## 🚀 Features
- Live crash point data fetching (Selenium)
- Prediction using Random Forest / XGBoost
- Telegram alerts for high predictions
- Secure login with password authentication
- Auto-refresh and live countdown

## 🛠 Tech Stack
- Python + Streamlit
- Scikit-learn, XGBoost
- Selenium for web scraping
- Telegram Bot API
- Streamlit Authenticator (secure admin login)

## 📦 Installation

```bash
pip install -r requirements.txt
```

To run locally:
```bash
streamlit run app.py
```

## 🌐 Deployment

1. Push this project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy the `app.py`
4. Add secrets (`Settings > Secrets`):

```toml
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

## 🤖 Telegram Bot Setup

1. Create a bot with [@BotFather](https://t.me/BotFather)
2. Use `/newbot` and get the API token
3. Use `getUpdates` to get your chat ID
4. Add them to `st.secrets`

## 🔒 Security
- Admin login is password protected using SHA-hashed passwords
- Chat alerts are private via Telegram
