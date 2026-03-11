import hashlib
import json
import os
from time import time
from datetime import datetime

class PredictionBlockchain:
    def __init__(self, chain_file="prediction_chain.json"):
        self.chain = []
        self.chain_file = chain_file
        self.load_chain()
        
        if not self.chain:
            self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_data = {
            "type": "genesis",
            "message": "Crypto AI Prediction System Genesis Block",
            "system": "Hybrid LSTM+GRU+CNN+Transformer+XGBoost with News Sentiment",
            "created": datetime.now().isoformat()
        }
        self.create_block(genesis_data, "0")

    def hash(self, block):
        """Create SHA-256 hash of a block"""
        block_dict = {k: v for k, v in block.items() if k != "hash"}
        encoded = json.dumps(block_dict, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def create_block(self, data, previous_hash):
        """Create a new block and add to chain"""
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time(),
            "timestamp_iso": datetime.now().isoformat(),
            "data": data,
            "previous_hash": previous_hash,
            "hash": ""
        }
        
        block["hash"] = self.hash(block)
        self.chain.append(block)
        self.save_chain()
        
        return block

    def add_prediction(self, symbol, current_price, predictions, sentiment_data, final_prediction):
        """
        Log a complete prediction with all model results and sentiment
        """
        last_block = self.chain[-1]
        previous_hash = last_block["hash"]
        
        block_data = {
            "type": "prediction",
            "symbol": symbol,
            "current_price": float(current_price),
            "individual_models": {
                "lstm": float(predictions.get('lstm', 0)),
                "gru": float(predictions.get('gru', 0)),
                "cnn_lstm": float(predictions.get('cnn', 0)),
                "transformer": float(predictions.get('transformer', 0)),
                "temporal_fusion": float(predictions.get('fusion', 0)),
                "xgboost": float(predictions.get('xgb', 0))
            },
            "sentiment_analysis": {
                "news_sentiment_score": float(sentiment_data.get('news_score', 0)),
                "news_sentiment_label": sentiment_data.get('news_label', 'Neutral'),
                "news_sentiment_percent": int(sentiment_data.get('news_percent', 50)),
                "technical_sentiment_score": float(sentiment_data.get('tech_score', 0)),
                "technical_sentiment_label": sentiment_data.get('tech_label', 'Neutral'),
                "combined_sentiment_score": float(sentiment_data.get('combined_score', 0))
            },
            "final_prediction": float(final_prediction),
            "expected_change_percent": float(((final_prediction - current_price) / current_price * 100)) if current_price != 0 else 0
        }
        
        return self.create_block(block_data, previous_hash)

    def verify_chain(self):
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current["hash"] != self.hash(current):
                return False, f"Invalid hash at block {i}"
            
            if current["previous_hash"] != previous["hash"]:
                return False, f"Chain broken at block {i}"
        
        return True, "Chain valid"

    def get_predictions_for_symbol(self, symbol, limit=10):
        """Get prediction history for a specific cryptocurrency"""
        predictions = []
        for block in reversed(self.chain):
            if block["data"].get("type") == "prediction" and block["data"].get("symbol") == symbol:
                predictions.append(block)
                if len(predictions) >= limit:
                    break
        return predictions

    def get_chain_stats(self):
        """Get blockchain statistics"""
        total_blocks = len(self.chain)
        prediction_blocks = sum(1 for b in self.chain if b["data"].get("type") == "prediction")
        
        return {
            "total_blocks": total_blocks,
            "prediction_blocks": prediction_blocks,
            "genesis_block": self.chain[0]["hash"] if self.chain else None,
            "last_block_hash": self.chain[-1]["hash"] if self.chain else None,
            "chain_valid": self.verify_chain()[0]
        }

    def save_chain(self):
        """Save chain to file"""
        try:
            with open(self.chain_file, 'w') as f:
                json.dump(self.chain, f, indent=2)
        except Exception as e:
            print(f"Error saving chain: {e}")

    def load_chain(self):
        """Load chain from file"""
        if os.path.exists(self.chain_file):
            try:
                with open(self.chain_file, 'r') as f:
                    self.chain = json.load(f)
            except Exception as e:
                print(f"Error loading chain: {e}")
                self.chain = []

    def show_chain(self):
        """Display the entire chain"""
        return self.chain

    def export_to_csv(self, filename="prediction_history.csv"):
        """Export predictions to CSV for analysis"""
        import csv
        
        predictions = [b for b in self.chain if b["data"].get("type") == "prediction"]
        
        if not predictions:
            print("No predictions to export")
            return
        
        with open(filename, 'w', newline='') as f:
            headers = ["index", "timestamp_iso", "symbol", "current_price", "final_prediction", 
                      "expected_change_percent", "news_sentiment_score", "news_sentiment_label",
                      "technical_sentiment_score", "lstm_pred", "gru_pred", "cnn_pred", 
                      "transformer_pred", "fusion_pred", "xgb_pred"]
            
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for block in predictions:
                data = block["data"]
                row = {
                    "index": block["index"],
                    "timestamp_iso": data.get("timestamp_iso", ""),
                    "symbol": data.get("symbol", ""),
                    "current_price": data.get("current_price", ""),
                    "final_prediction": data.get("final_prediction", ""),
                    "expected_change_percent": data.get("expected_change_percent", ""),
                    "news_sentiment_score": data["sentiment_analysis"].get("news_sentiment_score", ""),
                    "news_sentiment_label": data["sentiment_analysis"].get("news_sentiment_label", ""),
                    "technical_sentiment_score": data["sentiment_analysis"].get("technical_sentiment_score", ""),
                    "lstm_pred": data["individual_models"].get("lstm", ""),
                    "gru_pred": data["individual_models"].get("gru", ""),
                    "cnn_pred": data["individual_models"].get("cnn_lstm", ""),
                    "transformer_pred": data["individual_models"].get("transformer", ""),
                    "fusion_pred": data["individual_models"].get("temporal_fusion", ""),
                    "xgb_pred": data["individual_models"].get("xgboost", "")
                }
                writer.writerow(row)
        
        print(f"Exported {len(predictions)} predictions to {filename}")


if __name__ == "__main__":
    # Test the blockchain
    bc = PredictionBlockchain()
    
    test_predictions = {
        'lstm': 45000.5,
        'gru': 45200.3,
        'cnn': 44800.7,
        'transformer': 45100.2,
        'fusion': 45300.1,
        'xgb': 44950.0
    }
    
    test_sentiment = {
        'news_score': 0.35,
        'news_label': 'Bullish 📈',
        'news_percent': 67,
        'tech_score': 0.15,
        'tech_label': 'Neutral ➡️',
        'combined_score': 0.25
    }
    
    block = bc.add_prediction("BTC-USD", 44000.0, test_predictions, test_sentiment, 45200.0)
    print(f"Added block #{block['index']} with hash: {block['hash'][:16]}...")
    print(f"Chain Stats: {bc.get_chain_stats()}")