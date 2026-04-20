import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period="10y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        
        if df is None or df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df["Stock"] = ticker
        
        return df
        
    except Exception as e:
        return None