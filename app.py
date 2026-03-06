# ==========================================
# 🚀 ADVANCED CRYPTO AI PREDICTOR + NEWS AI
# WITH ORIGINAL vs PREDICTED GRAPH
# ==========================================

import os
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import traceback

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    raise ValueError("❌ NEWS_API_KEY not found in .env file")

newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ================= CONFIG =================
WINDOW = 60

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "EMA_10", "EMA_30", "RSI", "MACD",
    "BB_High", "BB_Low", "ATR"
]

# ================= LOAD MODELS =================
print("🔄 Loading AI models...")

try:
    lstm = load_model("models/lstm.keras")
    gru = load_model("models/gru.keras")
    cnn = load_model("models/cnn_lstm.keras")

    xgb = joblib.load("models/xgb.pkl")
    meta_model = joblib.load("models/meta.pkl")
    scaler = joblib.load("models/scaler.pkl")

    print("✅ All models loaded successfully!")

except Exception as e:
    print("❌ Error loading models")
    print(e)

# ================= FLASK APP =================
app = Flask(__name__)

# ================= SENTIMENT FUNCTION =================
def get_sentiment(symbol):
    try:
        news = newsapi.get_everything(
            q=symbol,
            language="en",
            sort_by="publishedAt",
            page_size=20
        )

        articles = news.get("articles", [])

        if not articles:
            return 0

        scores = []

        for article in articles:

            text = (
                (article.get("title", "") or "") +
                " " +
                (article.get("description", "") or "")
            )

            blob = TextBlob(text)
            scores.append(blob.sentiment.polarity)

        return round(float(np.mean(scores)), 3)

    except Exception:
        return 0


# ================= MAIN ROUTE =================
@app.route("/", methods=["GET", "POST"])
def home():

    try:

        # ========= USER INPUT =========
        if request.method == "POST":
            SYMBOL = request.form.get("stock", "BTC-USD").upper().strip()
            DAYS_TO_PREDICT = int(request.form.get("no_of_days", 10))
        else:
            SYMBOL = "BTC-USD"
            DAYS_TO_PREDICT = 10

        # ========= DOWNLOAD DATA =========
        df = yf.download(SYMBOL, period="3y")

        if df is None or df.empty:
            return render_template("index.html",
                                   error="❌ Failed to fetch crypto data")

        if len(df) < 100:
            return render_template("index.html",
                                   error="❌ Not enough historical data")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < WINDOW:
            return render_template("index.html",
                                   error="❌ Not enough data")

        close = df["Close"].squeeze()

        # ========= TECHNICAL INDICATORS =========
        df["EMA_10"] = ta.trend.ema_indicator(close, 10)
        df["EMA_30"] = ta.trend.ema_indicator(close, 30)
        df["RSI"] = ta.momentum.rsi(close, 14)
        df["MACD"] = ta.trend.macd(close)

        df["BB_High"] = ta.volatility.bollinger_hband(close)
        df["BB_Low"] = ta.volatility.bollinger_lband(close)

        df["ATR"] = ta.volatility.average_true_range(
            df["High"], df["Low"], close
        )

        df.dropna(inplace=True)

        if len(df) < WINDOW:
            return render_template("index.html",
                                   error="❌ Not enough processed data")

        data = df[FEATURES]

        # ========= SCALE =========
        scaled = scaler.transform(data)

        last_window = scaled[-WINDOW:]
        current_window = last_window.copy()

        future_prices = []

        # ========= PREDICTION LOOP =========
        for _ in range(DAYS_TO_PREDICT):

            X_input = np.expand_dims(current_window, axis=0)

            lstm_p = lstm.predict(X_input, verbose=0)[0][0]
            gru_p = gru.predict(X_input, verbose=0)[0][0]
            cnn_p = cnn.predict(X_input, verbose=0)[0][0]

            xgb_p = xgb.predict(current_window.reshape(1, -1))[0]

            meta_input = np.array([[lstm_p, gru_p, cnn_p, xgb_p]])

            scaled_pred = meta_model.predict(meta_input)[0]

            dummy = np.zeros((1, len(FEATURES)))

            dummy[0, FEATURES.index("Close")] = scaled_pred

            real_price = scaler.inverse_transform(dummy)[0][
                FEATURES.index("Close")
            ]

            future_prices.append(round(float(real_price), 2))

            new_row = current_window[-1].copy()

            new_row[FEATURES.index("Close")] = scaled_pred

            current_window = np.vstack([current_window[1:], new_row])

        current_real_price = round(float(df["Close"].iloc[-1]), 2)

        # ========= SENTIMENT =========
        sentiment_score = get_sentiment(SYMBOL)

        if sentiment_score > 0:
            sentiment_label = "🟢 Positive"
        elif sentiment_score < 0:
            sentiment_label = "🔴 Negative"
        else:
            sentiment_label = "🟡 Neutral"

        # ========= GRAPH DATA =========
        real_prices = df["Close"].tail(60).tolist()

        predicted_line = (
            [None] * (len(real_prices) - 1)
            + [real_prices[-1]]
            + future_prices
        )

        labels = list(range(len(real_prices) + len(future_prices)))

        # ========= RENDER =========
        return render_template(
            "index.html",
            symbol=SYMBOL,
            current_price=current_real_price,
            future=future_prices,
            sentiment=sentiment_score,
            sentiment_label=sentiment_label,
            real_prices=real_prices,
            predicted_line=predicted_line,
            labels=labels
        )

    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", error=str(e))


# ================= RUN =================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )