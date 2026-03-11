# ==========================================
# ADVANCED CRYPTO AI PREDICTOR - FINAL APP
# With News API Sentiment + Blockchain Logging
# ==========================================

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import requests
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

from blockchain import PredictionBlockchain

app = Flask(__name__)

# ================= LOAD MODELS =================

try:
    lstm = load_model("model_lstm.keras")
    gru = load_model("model_gru.keras")
    cnn = load_model("model_cnn_lstm.keras")
    transformer = load_model("model_transformer.keras")
    fusion = load_model("model_temporal_fusion.keras")
    xgb = joblib.load("xgb_model.pkl")
    meta_model = joblib.load("meta_model.pkl")
    scaler = joblib.load("scaler.save")
    config = joblib.load("model_config.pkl")
    FEATURES = config['features']
    MODELS_LOADED = True
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Models not loaded - {e}")
    MODELS_LOADED = False
    FEATURES = [
        "Open","High","Low","Close","Volume",
        "EMA_10","EMA_30","RSI","MACD",
        "BB_High","BB_Low","ATR",
        "Momentum","ROC","CCI","ADX",
        "News_Sentiment", "Tech_Sentiment"
    ]

WINDOW = 90
NEWS_API_KEY = "24114dded3054afeb024157b3ddb3f11"  # Replace with your actual API key

# Initialize Blockchain
blockchain = PredictionBlockchain()

# Import TA functions correctly
from ta.momentum import rsi as ta_rsi, roc as ta_roc
from ta.trend import ema_indicator as ta_ema, macd as ta_macd, cci as ta_cci, adx as ta_adx
from ta.volatility import bollinger_hband as ta_bb_high, bollinger_lband as ta_bb_low, average_true_range as ta_atr

# ================= NEWS API SENTIMENT =================

class NewsAPISentiment:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def fetch_news(self, query="Bitcoin cryptocurrency", days_back=3):
        """Fetch news from NewsAPI with caching"""
        cache_key = f"{query}_{days_back}"
        current_time = time.time()
        
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if current_time - cached_time < self.cache_duration:
                return cached_data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                self.cache[cache_key] = (articles, current_time)
                return articles
            else:
                print(f"News API Error: {data.get('message')}")
                return []
        except Exception as e:
            print(f"News fetch error: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using keyword matching"""
        text = text.lower()
        
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'moon', 
                         'breakout', 'pump', 'adoption', 'partnership', 'growth',
                         'strong', 'positive', 'optimistic', 'record', 'high',
                         'soar', 'boost', 'recover', 'support', 'buy', 'bull']
        
        negative_words = ['bearish', 'crash', 'drop', 'fall', 'down', 'dump', 'fud',
                         'fear', 'sell', 'loss', 'decline', 'bear', 'correction',
                         'low', 'weak', 'negative', 'pessimistic', 'ban', 'regulation',
                         'hack', 'scam', 'fraud', 'risk', 'volatile', 'crash']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
        
        sentiment = (pos_count - neg_count) / total
        return max(-1, min(1, sentiment))
    
    def get_sentiment(self, symbol="Bitcoin"):
        """Get comprehensive sentiment analysis"""
        search_terms = {
            "BTC-USD": "Bitcoin BTC",
            "ETH-USD": "Ethereum ETH",
            "SOL-USD": "Solana SOL",
            "ADA-USD": "Cardano ADA",
            "DOT-USD": "Polkadot DOT",
            "XRP-USD": "Ripple XRP",
            "LTC-USD": "Litecoin LTC"
        }
        
        query = search_terms.get(symbol, f"{symbol.split('-')[0]} cryptocurrency")
        articles = self.fetch_news(query)
        
        if not articles:
            return {
                'score': 0,
                'percent': 50,
                'label': 'Neutral ➡️',
                'article_count': 0,
                'articles': []
            }
        
        sentiments = []
        analyzed_articles = []
        
        for article in articles[:15]:
            title = article.get('title', '')
            desc = article.get('description', '') or ''
            content = f"{title} {desc}"
            
            sentiment = self.analyze_sentiment(content)
            sentiments.append(sentiment)
            
            analyzed_articles.append({
                'title': title[:80] + "..." if len(title) > 80 else title,
                'source': article.get('source', {}).get('name', 'Unknown'),
                'sentiment': round(sentiment, 2),
                'published': article.get('publishedAt', '')[:10]
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
        
        return {
            'score': round(avg_sentiment, 3),
            'percent': sentiment_percent,
            'label': label,
            'article_count': len(articles),
            'articles': analyzed_articles[:5]
        }

# Initialize News API
news_api = NewsAPISentiment(NEWS_API_KEY)

# ================= TECHNICAL SENTIMENT =================

def get_technical_sentiment(df):
    """Calculate technical indicator based sentiment"""
    try:
        close = df['Close']
        
        rsi = ta_rsi(close, window=14).iloc[-1]
        if pd.isna(rsi):
            rsi = 50
        
        macd = ta_macd(close).iloc[-1]
        macd_signal = ta.trend.macd_signal(close).iloc[-1]
        
        ema20 = ta_ema(close, window=20).iloc[-1]
        price = close.iloc[-1]
        
        score = 0
        
        # RSI contribution
        if rsi > 70:
            score += 0.4
        elif rsi < 30:
            score -= 0.4
        else:
            score += ((rsi - 50) / 20) * 0.4
        
        # MACD contribution
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal:
                score += 0.3
            else:
                score -= 0.3
        
        # Price vs EMA contribution
        if not pd.isna(ema20):
            if price > ema20:
                score += 0.3
            else:
                score -= 0.3
        
        score = max(-1, min(1, score))
        percent = int((score + 1) / 2 * 100)
        
        if score > 0.3:
            label = "Very Bullish 🚀"
        elif score > 0.1:
            label = "Bullish 📈"
        elif score > -0.1:
            label = "Neutral ➡️"
        elif score > -0.3:
            label = "Bearish 📉"
        else:
            label = "Very Bearish 🔻"
        
        return {
            'score': round(score, 3),
            'percent': percent,
            'label': label,
            'indicators': {
                'rsi': round(rsi, 2),
                'macd': round(macd, 4) if not pd.isna(macd) else 0,
                'price_vs_ema20': round((price/ema20 - 1)*100, 2) if not pd.isna(ema20) else 0
            }
        }
    except Exception as e:
        print(f"Technical sentiment error: {e}")
        return {'score': 0, 'percent': 50, 'label': 'Neutral ➡️', 'indicators': {}}

# ================= PREPARE DATA =================

def prepare_data(symbol):
    """Prepare data with all features including sentiment"""
    df = yf.download(symbol, period="5y", progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {symbol}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df["Close"]
    
    # Technical indicators
    df["EMA_10"] = ta_ema(close, window=10)
    df["EMA_30"] = ta_ema(close, window=30)
    df["RSI"] = ta_rsi(close, window=14)
    df["MACD"] = ta_macd(close)
    df["BB_High"] = ta_bb_high(close)
    df["BB_Low"] = ta_bb_low(close)
    df["ATR"] = ta_atr(df["High"], df["Low"], close, window=14)
    df["Momentum"] = close.diff()
    df["ROC"] = ta_roc(close)
    df["CCI"] = ta_cci(df["High"], df["Low"], close, window=20)
    df["ADX"] = ta_adx(df["High"], df["Low"], close, window=14)
    
    # Get sentiments
    news_sent = news_api.get_sentiment(symbol)
    tech_sent = get_technical_sentiment(df)
    
    # Add sentiment features
    df['News_Sentiment'] = news_sent['score']
    df['Tech_Sentiment'] = tech_sent['score']
    
    df.dropna(inplace=True)
    data = df[FEATURES]
    scaled = scaler.transform(data)
    
    X = [scaled[i-WINDOW:i] for i in range(WINDOW, len(scaled))]
    
    return df, np.array(X), data, news_sent, tech_sent

# ================= PREDICTION =================

def inverse_transform_single(value, feature_index):
    """Helper to inverse transform a single value"""
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, feature_index] = value
    return scaler.inverse_transform(dummy)[0, feature_index]

def predict_price(symbol, days):
    """Make prediction with all models and sentiment integration"""
    if not MODELS_LOADED:
        raise RuntimeError("Models not loaded. Please train and save models first.")
    
    df, X, data, news_sent, tech_sent = prepare_data(symbol)
    X_last = X[-1:]
    
    # Individual model predictions (scaled)
    lstm_pred_scaled = float(lstm.predict(X_last, verbose=0)[0][0])
    gru_pred_scaled = float(gru.predict(X_last, verbose=0)[0][0])
    cnn_pred_scaled = float(cnn.predict(X_last, verbose=0)[0][0])
    transformer_pred_scaled = float(transformer.predict(X_last, verbose=0)[0][0])
    fusion_pred_scaled = float(fusion.predict(X_last, verbose=0)[0][0])
    
    X_flat = X_last.reshape(1, -1)
    xgb_pred_scaled = float(xgb.predict(X_flat)[0])
    
    # Convert to actual prices
    close_index = FEATURES.index("Close")
    lstm_price = float(inverse_transform_single(lstm_pred_scaled, close_index))
    gru_price = float(inverse_transform_single(gru_pred_scaled, close_index))
    cnn_price = float(inverse_transform_single(cnn_pred_scaled, close_index))
    transformer_price = float(inverse_transform_single(transformer_pred_scaled, close_index))
    fusion_price = float(inverse_transform_single(fusion_pred_scaled, close_index))
    xgb_price = float(inverse_transform_single(xgb_pred_scaled, close_index))
    
    # Meta ensemble with sentiment
    combined_sentiment = (news_sent['score'] + tech_sent['score']) / 2
    
    meta_input = np.array([[
        lstm_pred_scaled, gru_pred_scaled, cnn_pred_scaled,
        transformer_pred_scaled, fusion_pred_scaled, xgb_pred_scaled,
        combined_sentiment
    ]])
    
    pred_scaled = float(meta_model.predict(meta_input)[0])
    predicted_price = float(inverse_transform_single(pred_scaled, close_index))
    current_price = float(df["Close"].iloc[-1])
    
    # Future predictions with sentiment bias
    future_prices = []
    price = predicted_price
    trend_bias = combined_sentiment * 0.002
    
    for i in range(days):
        random_change = np.random.normal(trend_bias, 0.015)
        price = price * (1 + random_change)
        future_prices.append(round(price, 2))
    
    # Chart data
    labels = list(range(60))
    real_prices = df["Close"].tail(60).tolist()
    predicted_line = [None] * 59 + [predicted_price]
    
    # Individual predictions dict
    individual_preds = {
        'lstm': lstm_price,
        'gru': gru_price,
        'cnn': cnn_price,
        'transformer': transformer_price,
        'fusion': fusion_price,
        'xgb': xgb_price
    }
    
    # Log to blockchain
    sentiment_data = {
        'news_score': news_sent['score'],
        'news_label': news_sent['label'],
        'news_percent': news_sent['percent'],
        'tech_score': tech_sent['score'],
        'tech_label': tech_sent['label'],
        'combined_score': combined_sentiment
    }
    
    blockchain.add_prediction(
        symbol=symbol,
        current_price=current_price,
        predictions=individual_preds,
        sentiment_data=sentiment_data,
        final_prediction=predicted_price
    )
    
    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "future_prices": future_prices,
        "news_sentiment": news_sent,
        "technical_sentiment": tech_sent,
        "combined_sentiment": {
            'score': round(combined_sentiment, 3),
            'percent': int((combined_sentiment + 1) / 2 * 100)
        },
        "individual_predictions": individual_preds,
        "labels": labels,
        "real_prices": real_prices,
        "predicted_line": predicted_line
    }

# ================= ROUTES =================

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    
    if request.method == "POST":
        try:
            symbol = request.form.get("stock", "").upper().strip()
            days = int(request.form.get("no_of_days", 7))
            
            if not symbol:
                raise ValueError("Please enter a valid crypto symbol (e.g., BTC-USD)")
            
            if days < 1 or days > 30:
                raise ValueError("Days must be between 1 and 30")
            
            result = predict_price(symbol, days)
            
        except Exception as e:
            error = str(e)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    return render_template("index.html", result=result, error=error)

@app.route("/api/predict/<symbol>")
def api_predict(symbol):
    """API endpoint for predictions"""
    try:
        days = int(request.args.get("days", 7))
        result = predict_price(symbol.upper(), days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/blockchain")
def api_blockchain():
    """Get blockchain info"""
    return jsonify({
        "stats": blockchain.get_chain_stats(),
        "recent_predictions": blockchain.get_predictions_for_symbol("BTC-USD", 5)
    })

@app.route("/api/blockchain/<symbol>")
def api_blockchain_symbol(symbol):
    """Get blockchain predictions for specific symbol"""
    return jsonify(blockchain.get_predictions_for_symbol(symbol.upper(), 10))

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)