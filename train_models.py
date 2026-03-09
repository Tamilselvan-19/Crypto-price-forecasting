# ==========================================
# FINAL HYBRID CRYPTO AI TRAINING SYSTEM
# LSTM + GRU + CNN-LSTM + Transformer
# Temporal Attention Fusion + XGBoost
# ==========================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import math
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    f1_score
)

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout,
    Conv1D, MaxPooling1D,
    LayerNormalization,
    MultiHeadAttention,
    Input,
    GlobalAveragePooling1D,
    Attention,
    Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ================= CONFIG =================

SYMBOL = "BTC-USD"
WINDOW = 90
EPOCHS = 5
BATCH_SIZE = 32


# ================= DOWNLOAD DATA =================

print("Downloading data...")

df = yf.download(SYMBOL, start="2015-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df["Close"].squeeze()


# ================= TECHNICAL INDICATORS =================

df["EMA_10"] = ta.trend.ema_indicator(close, 10)
df["EMA_30"] = ta.trend.ema_indicator(close, 30)
df["RSI"] = ta.momentum.rsi(close, 14)
df["MACD"] = ta.trend.macd(close)

df["BB_High"] = ta.volatility.bollinger_hband(close)
df["BB_Low"] = ta.volatility.bollinger_lband(close)
df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], close)

# Momentum indicators
df["Momentum"] = close.diff()
df["ROC"] = ta.momentum.roc(close)
df["CCI"] = ta.trend.cci(df["High"], df["Low"], close)
df["ADX"] = ta.trend.adx(df["High"], df["Low"], close)

df.dropna(inplace=True)


FEATURES = [
    "Open","High","Low","Close","Volume",
    "EMA_10","EMA_30","RSI","MACD",
    "BB_High","BB_Low","ATR",
    "Momentum","ROC","CCI","ADX"
]

data = df[FEATURES]


# ================= SCALING =================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, y = [], []

for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])
    y.append(scaled[i, FEATURES.index("Close")])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.9)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# ================= MODEL DEFINITIONS =================

def build_lstm():

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, X.shape[2])),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


def build_gru():

    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(WINDOW, X.shape[2])),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


def build_cnn_lstm():

    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(WINDOW, X.shape[2])),
        MaxPooling1D(2),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


# ================= TRANSFORMER =================

def build_transformer():

    inputs = Input(shape=(WINDOW, X.shape[2]))

    attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)

    x = LayerNormalization()(attention + inputs)

    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


# ================= TEMPORAL ATTENTION FUSION =================

def build_temporal_fusion():

    inputs = Input(shape=(WINDOW, X.shape[2]))

    lstm_out = LSTM(64, return_sequences=True)(inputs)
    gru_out = GRU(64, return_sequences=True)(inputs)

    combined = Concatenate()([lstm_out, gru_out])

    attention = Attention()([combined, combined])

    x = GlobalAveragePooling1D()(attention)

    x = Dense(64, activation="relu")(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(0.0005), loss="mse")

    return model


# ================= TRAIN MODELS =================

early = EarlyStopping(patience=7, restore_best_weights=True)

print("Training LSTM...")
lstm = build_lstm()
lstm.fit(X_train, y_train, epochs=EPOCHS,
         batch_size=BATCH_SIZE,
         validation_data=(X_test,y_test),
         callbacks=[early])

print("Training GRU...")
gru = build_gru()
gru.fit(X_train, y_train, epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test,y_test),
        callbacks=[early])

print("Training CNN-LSTM...")
cnn = build_cnn_lstm()
cnn.fit(X_train, y_train, epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test,y_test),
        callbacks=[early])

print("Training Transformer...")
transformer = build_transformer()
transformer.fit(X_train, y_train, epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test,y_test),
                callbacks=[early])

print("Training Temporal Fusion Model...")
fusion = build_temporal_fusion()
fusion.fit(X_train, y_train, epochs=EPOCHS,
           batch_size=BATCH_SIZE,
           validation_data=(X_test,y_test),
           callbacks=[early])


# ================= XGBOOST =================

X_flat = X.reshape(X.shape[0], -1)

print("Training XGBoost...")

xgb = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03
)

xgb.fit(X_flat[:split], y_train)


# ================= STACKING =================

print("Training Meta Model...")

lstm_pred = lstm.predict(X_test).flatten()
gru_pred = gru.predict(X_test).flatten()
cnn_pred = cnn.predict(X_test).flatten()
transformer_pred = transformer.predict(X_test).flatten()
fusion_pred = fusion.predict(X_test).flatten()
xgb_pred = xgb.predict(X_flat[split:])

meta_X = np.column_stack(
    (lstm_pred, gru_pred, cnn_pred,
     transformer_pred, fusion_pred, xgb_pred)
)

meta_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=4
)

meta_model.fit(meta_X, y_test)

predictions = meta_model.predict(meta_X)


# ================= EVALUATION =================

rmse = math.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

actual_direction = np.sign(np.diff(y_test))
pred_direction = np.sign(np.diff(predictions))

directional_accuracy = np.mean(actual_direction == pred_direction)

actual_class = (actual_direction > 0).astype(int)
pred_class = (pred_direction > 0).astype(int)

f1 = f1_score(actual_class, pred_class)


print("\n================ MODEL PERFORMANCE ================")
print("RMSE :", rmse)
print("MAE  :", mae)
print("MAPE :", mape)
print("R2   :", r2)
print("Directional Accuracy :", directional_accuracy)
print("F1 Score :", f1)
print("==================================================")


# ================= SAVE MODELS =================

print("Saving models...")

lstm.save("model_lstm.keras")
gru.save("model_gru.keras")
cnn.save("model_cnn_lstm.keras")
transformer.save("model_transformer.keras")
fusion.save("model_temporal_fusion.keras")

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(scaler, "scaler.save")

print("ALL MODELS SAVED")