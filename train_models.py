# ==========================================
# ADVANCED CRYPTO AI TRAINING (UPGRADED)
# ==========================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import math
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ================= CONFIG =================
SYMBOL = "BTC-USD"
WINDOW = 60
EPOCHS = 25
BATCH_SIZE = 32

# ================= DOWNLOAD DATA =================
df = yf.download(SYMBOL, start="2015-01-01")

# Fix: Flatten MultiIndex columns after yfinance download
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df["Close"].squeeze()

# ================= TECHNICAL INDICATORS =================
df["EMA_10"] = ta.trend.ema_indicator(close, 10)
df["EMA_30"] = ta.trend.ema_indicator(close, 30)
df["RSI"] = ta.momentum.rsi(close, 14)
df["MACD"] = ta.trend.macd(close)

# 🔥 NEW FEATURES
df["BB_High"] = ta.volatility.bollinger_hband(close)
df["BB_Low"] = ta.volatility.bollinger_lband(close)
df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], close)

df.dropna(inplace=True)

FEATURES = [
    "Open","High","Low","Close","Volume",
    "EMA_10","EMA_30","RSI","MACD",
    "BB_High","BB_Low","ATR"
]

data = df[FEATURES]

# ================= SCALING =================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])
    y.append(scaled[i, FEATURES.index("Close")])

X, y = np.array(X), np.array(y)

split = int(len(X)*0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ================= MODELS =================
def build_lstm():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru():
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(WINDOW, X.shape[2])),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_cnn_lstm():
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(WINDOW, X.shape[2])),
        MaxPooling1D(2),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

early = EarlyStopping(patience=5, restore_best_weights=True)

lstm = build_lstm()
lstm.fit(X_train, y_train, epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         validation_data=(X_test,y_test),
         callbacks=[early])

gru = build_gru()
gru.fit(X_train, y_train, epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test,y_test),
        callbacks=[early])

cnn = build_cnn_lstm()
cnn.fit(X_train, y_train, epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test,y_test),
        callbacks=[early])

# ================= XGBOOST =================
X_flat = X.reshape(X.shape[0], -1)
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb.fit(X_flat[:split], y_train)

# ================= STACKING =================
lstm_pred = lstm.predict(X_test).flatten()
gru_pred = gru.predict(X_test).flatten()
cnn_pred = cnn.predict(X_test).flatten()
xgb_pred = xgb.predict(X_flat[split:])

meta_X = np.column_stack((lstm_pred, gru_pred, cnn_pred, xgb_pred))
meta_model = LinearRegression()
meta_model.fit(meta_X, y_test)

rmse = math.sqrt(mean_squared_error(y_test, meta_model.predict(meta_X)))
print("Final RMSE:", rmse)

# ================= SAVE =================
lstm.save("model_lstm.keras")
gru.save("model_gru.keras")
cnn.save("model_cnn_lstm.keras")

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(scaler, "scaler.save")

print("✅ ALL MODELS SAVED")
