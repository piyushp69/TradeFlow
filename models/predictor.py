import joblib
import os
import pandas as pd

class StockPredictor:
    def __init__(self):
        self.features = [
            "Daily_Return", 
            "RSI", 
            "Volatility", 
            "SMA20_Ratio", 
            "SMA50_Ratio", 
            "MACD_Ratio"
        ]
        self.models = {}

        periods = ['weekly', 'monthly', 'yearly']
        for period in periods:
            model_path = os.path.join("models", "saved_models", f"model_{period}.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[period] = joblib.load(model_path)
                except Exception:
                    pass

    def predict_all(self, current_data):
        if not self.models:
            return None
        
        try:
            last_row = current_data[self.features].tail(1)
            
            if last_row.isnull().values.any():
                return None
                
            results = {}
            for period, model in self.models.items():
                probs = model.predict_proba(last_row)[0]
                up_prob = probs[1]

                if period == 'weekly':
                    threshold = 0.48
                elif period == 'yearly':
                    threshold = 0.55
                else:
                    threshold = 0.50 

                if up_prob >= threshold:
                    direction = "UP"
                    conf = round(up_prob * 100, 2)
                else:
                    direction = "DOWN"
                    conf = round((1 - up_prob) * 100, 2)
                
                results[period] = (direction, conf)
                
            return results
            
        except Exception:
            return None