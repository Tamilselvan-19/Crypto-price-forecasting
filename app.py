# ==========================================
# ADVANCED CRYPTO AI PREDICTOR - FINAL APP
# ==========================================

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================= LOAD MODELS =================

lstm = load_model("model_lstm.keras")
gru = load_model("model_gru.keras")
cnn = load_model("model_cnn_lstm.keras")
transformer = load_model("model_transformer.keras")
fusion = load_model("model_temporal_fusion.keras")

xgb = joblib.load("xgb_model.pkl")
meta_model = joblib.load("meta_model.pkl")
scaler = joblib.load("scaler.save")

WINDOW = 90

FEATURES = [
"Open","High","Low","Close","Volume",
"EMA_10","EMA_30","RSI","MACD",
"BB_High","BB_Low","ATR",
"Momentum","ROC","CCI","ADX"
]


# ================= PREPARE DATA =================

def prepare_data(symbol):

    df = yf.download(symbol, period="5y")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    df["EMA_10"] = ta.trend.ema_indicator(close, 10)
    df["EMA_30"] = ta.trend.ema_indicator(close, 30)

    df["RSI"] = ta.momentum.rsi(close, 14)
    df["MACD"] = ta.trend.macd(close)

    df["BB_High"] = ta.volatility.bollinger_hband(close)
    df["BB_Low"] = ta.volatility.bollinger_lband(close)

    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], close)

    df["Momentum"] = close.diff()
    df["ROC"] = ta.momentum.roc(close)
    df["CCI"] = ta.trend.cci(df["High"], df["Low"], close)
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], close)

    df.dropna(inplace=True)

    data = df[FEATURES]

    scaled = scaler.transform(data)

    X = []
    for i in range(WINDOW, len(scaled)):
        X.append(scaled[i-WINDOW:i])

    X = np.array(X)

    return df, X, data


# ================= PREDICTION =================

def predict_price(symbol, days):

    df, X, data = prepare_data(symbol)

    X_last = X[-1:]

    lstm_pred = lstm.predict(X_last)
    gru_pred = gru.predict(X_last)
    cnn_pred = cnn.predict(X_last)
    transformer_pred = transformer.predict(X_last)
    fusion_pred = fusion.predict(X_last)

    X_flat = X_last.reshape(1, -1)
    xgb_pred = xgb.predict(X_flat)

    meta_input = np.column_stack((
        lstm_pred.flatten(),
        gru_pred.flatten(),
        cnn_pred.flatten(),
        transformer_pred.flatten(),
        fusion_pred.flatten(),
        xgb_pred
    ))

    pred_scaled = meta_model.predict(meta_input)[0]

    close_index = FEATURES.index("Close")

    dummy = np.zeros((1,len(FEATURES)))
    dummy[0,close_index] = pred_scaled

    predicted_price = scaler.inverse_transform(dummy)[0,close_index]

    current_price = df["Close"].iloc[-1]

    # ===== FUTURE PREDICTIONS (simple extension) =====

    future_prices = []
    price = predicted_price

    for i in range(days):
        price = price * (1 + np.random.normal(0,0.002))
        future_prices.append(round(price,2))

    # ===== CHART DATA =====

    labels = list(range(len(df.tail(60))))
    real_prices = df["Close"].tail(60).tolist()

    predicted_line = [None]*(len(real_prices)-1) + [predicted_price]

    return {
        "current_price": round(current_price,2),
        "future_prices": future_prices,
        "sentiment": 0,
        "sentiment_label": "Neutral",
        "labels": labels,
        "real_prices": real_prices,
        "predicted_line": predicted_line
    }


# ================= ROUTE =================

@app.route("/", methods=["GET","POST"])
def index():

    error = None
    symbol = None
    current_price = None
    future = None
    sentiment = None
    sentiment_label = None
    labels = []
    real_prices = []
    predicted_line = []

    if request.method == "POST":

        try:

            symbol = request.form.get("stock")
            days = int(request.form.get("no_of_days"))

            result = predict_price(symbol, days)

            current_price = result["current_price"]
            future = result["future_prices"]
            sentiment = result["sentiment"]
            sentiment_label = result["sentiment_label"]
            labels = result["labels"]
            real_prices = result["real_prices"]
            predicted_line = result["predicted_line"]

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        symbol=symbol,
        current_price=current_price,
        future=future,
        sentiment=sentiment,
        sentiment_label=sentiment_label,
        labels=labels,
        real_prices=real_prices,
        predicted_line=predicted_line,
        error=error
    )


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)