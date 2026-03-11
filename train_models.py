# ==========================================
# FINAL HYBRID CRYPTO AI TRAINING SYSTEM
# LSTM + GRU + CNN-LSTM + Transformer + Temporal Fusion + XGBoost
# With News API Sentiment Integration
# ==========================================

import yfinance as yf
import pandas as pd
import numpy as np
import math
import joblib
import requests
import json
from datetime import datetime, timedelta
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, f1_score
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

# Import TA indicators correctly
from ta.momentum import rsi as ta_rsi, roc as ta_roc
from ta.trend import ema_indicator as ta_ema, macd as ta_macd, cci as ta_cci, adx as ta_adx
from ta.volatility import bollinger_hband as ta_bb_high, bollinger_lband as ta_bb_low, average_true_range as ta_atr

# ================= CONFIG =================

SYMBOL = "BTC-USD"
WINDOW = 90
EPOCHS = 1
BATCH_SIZE = 32
NEWS_API_KEY = "24114dded3054afeb024157b3ddb3f11"  # Get from newsapi.org

# ================= NEWS API SENTIMENT =================

class NewsSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_crypto_news(self, symbol="Bitcoin", days_back=7):
        """Fetch cryptocurrency news from NewsAPI"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = f"{symbol} OR cryptocurrency OR crypto OR blockchain"
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                print(f"News API Error: {data.get('message', 'Unknown error')}")
                return []
        except Exception as e:
            print(f"News fetch error: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis using keyword matching"""
        text = text.lower()
        
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'moon', 
                         'breakout', 'pump', 'adoption', 'partnership', 'growth',
                         'strong', 'positive', 'optimistic', 'record', 'high',
                         'soar', 'boost', 'recover', 'support', 'buy']
        
        negative_words = ['bearish', 'crash', 'drop', 'fall', 'down', 'dump', 'fud',
                         'fear', 'sell', 'loss', 'decline', 'bear', 'correction',
                         'low', 'weak', 'negative', 'pessimistic', 'ban', 'regulation',
                         'hack', 'scam', 'fraud', 'risk', 'volatile']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
        
        sentiment = (pos_count - neg_count) / total
        return max(-1, min(1, sentiment))
    
    def get_news_sentiment(self, symbol="Bitcoin"):
        """Get aggregated sentiment from recent news"""
        articles = self.fetch_crypto_news(symbol)
        
        if not articles:
            return 0, 50, "Neutral ➡️", []
        
        sentiments = []
        article_data = []
        
        for article in articles[:20]:
            title = article.get('title', '')
            description = article.get('description', '') or ''
            content = f"{title} {description}"
            
            sentiment = self.analyze_sentiment(content)
            sentiments.append(sentiment)
            
            article_data.append({
                'title': title,
                'source': article.get('source', {}).get('name', 'Unknown'),
                'sentiment': sentiment,
                'published': article.get('publishedAt', '')
            })
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        sentiment_percent = int((avg_sentiment + 1) / 2 * 100)
        
        if avg_sentiment > 0.3:
            label = "Very Bullish 🚀"
        elif avg_sentiment > 0.1:
            label = "Bullish 📈"
        elif avg_sentiment > -0.1:
            label = "Neutral ➡️"
        elif avg_sentiment > -0.3:
            label = "Bearish 📉"
        else:
            label = "Very Bearish 🔻"
        
        return avg_sentiment, sentiment_percent, label, article_data

# Initialize sentiment analyzer
news_analyzer = NewsSentimentAnalyzer(NEWS_API_KEY)

# ================= DOWNLOAD DATA =================

print("Downloading data...")

df = yf.download(SYMBOL, start="2015-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

close = df["Close"].squeeze()

# ================= TECHNICAL INDICATORS =================

print("Calculating technical indicators...")

df["EMA_10"] = ta_ema(close, window=10)
df["EMA_30"] = ta_ema(close, window=30)
df["RSI"] = ta_rsi(close, window=14)  # FIXED: Using ta.momentum.rsi
df["MACD"] = ta_macd(close)
df["BB_High"] = ta_bb_high(close)
df["BB_Low"] = ta_bb_low(close)
df["ATR"] = ta_atr(df["High"], df["Low"], close, window=14)
df["Momentum"] = close.diff()
df["ROC"] = ta_roc(close)
df["CCI"] = ta_cci(df["High"], df["Low"], close, window=20)
df["ADX"] = ta_adx(df["High"], df["Low"], close, window=14)

# Add News Sentiment Feature
print("Fetching news sentiment...")
news_sentiment, news_sent_percent, news_label, _ = news_analyzer.get_news_sentiment("Bitcoin")
print(f"News Sentiment: {news_label} ({news_sent_percent}%)")

df['News_Sentiment'] = news_sentiment
df['Tech_Sentiment'] = 0  # Placeholder, calculated per row in production

df.dropna(inplace=True)

# IMPORTANT: DO NOT CHANGE ORDER
FEATURES = [
    "Open","High","Low","Close","Volume",
    "EMA_10","EMA_30","RSI","MACD",
    "BB_High","BB_Low","ATR",
    "Momentum","ROC","CCI","ADX",
    "News_Sentiment", "Tech_Sentiment"
]

data = df[FEATURES]

# ================= SCALING =================

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X = []
y = []

for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])
    y.append(scaled[i, FEATURES.index("Close")])

X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.9)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

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

early = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

print("\nTraining LSTM...")
lstm = build_lstm()
lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
         validation_data=(X_test, y_test), callbacks=[early], verbose=1)

print("\nTraining GRU...")
gru = build_gru()
gru.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test), callbacks=[early], verbose=1)

print("\nTraining CNN-LSTM...")
cnn = build_cnn_lstm()
cnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test), callbacks=[early], verbose=1)

print("\nTraining Transformer...")
transformer = build_transformer()
transformer.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test), callbacks=[early], verbose=1)

print("\nTraining Temporal Fusion...")
fusion = build_temporal_fusion()
fusion.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
           validation_data=(X_test, y_test), callbacks=[early], verbose=1)

# ================= XGBOOST =================

print("\nTraining XGBoost...")
X_flat = X.reshape(X.shape[0], -1)

xgb = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    random_state=42
)

xgb.fit(X_flat[:split], y_train)

# ================= STACKING WITH SENTIMENT =================

print("\nTraining Meta Model with Sentiment Integration...")

lstm_pred = lstm.predict(X_test, verbose=0).flatten()
gru_pred = gru.predict(X_test, verbose=0).flatten()
cnn_pred = cnn.predict(X_test, verbose=0).flatten()
transformer_pred = transformer.predict(X_test, verbose=0).flatten()
fusion_pred = fusion.predict(X_test, verbose=0).flatten()
xgb_pred = xgb.predict(X_flat[split:])

# Include sentiment as additional feature for meta-learner
test_sentiment = np.full(len(X_test), news_sentiment)

meta_X = np.column_stack((
    lstm_pred, gru_pred, cnn_pred,
    transformer_pred, fusion_pred, xgb_pred,
    test_sentiment
))

meta_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=4,
    random_state=42
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

directional_accuracy = np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0

actual_class = (actual_direction > 0).astype(int)
pred_class = (pred_direction > 0).astype(int)

f1 = f1_score(actual_class, pred_class) if len(actual_class) > 0 else 0

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"MAPE : {mape:.6f}")
print(f"R2   : {r2:.6f} ({r2*100:.2f}%)")
print(f"Directional Accuracy : {directional_accuracy*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")
print("="*50)

# ================= SAVE MODELS =================

print("\nSaving models...")

lstm.save("model_lstm.keras")
gru.save("model_gru.keras")
cnn.save("model_cnn_lstm.keras")
transformer.save("model_transformer.keras")
fusion.save("model_temporal_fusion.keras")

joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(scaler, "scaler.save")

# Save feature list and config
config = {
    'features': FEATURES,
    'window': WINDOW,
    'news_sentiment_training': news_sentiment
}
joblib.dump(config, "model_config.pkl")

print("ALL MODELS SAVED SUCCESSFULLY")