import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from features.indicators import calculate_indicators
from sklearn.ensemble import GradientBoostingClassifier

STOCKS = [
    '^NSEI', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'SBIN.NS', 
    'BHARTIARTL.NS', 'ITC.NS', 'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 
    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'HCLTECH.NS', 'TATASTEEL.NS', 
    'NTPC.NS', 'POWERGRID.NS', 'KOTAKBANK.NS', 'ADANIENT.NS', 'COALINDIA.NS', 'BAJAJFINSV.NS',
    'ULTRACEMCO.NS', 'JSWSTEEL.NS', 'GRASIM.NS', 'M&M.NS', 'INDUSINDBK.NS', 'HINDALCO.NS', 
    'NESTLEIND.NS', 'ADANIPORTS.NS', 'ONGC.NS', 'TECHM.NS', 'WIPRO.NS', 'BRITANNIA.NS', 
    'EICHERMOT.NS', 'BPCL.NS', 'DIVISLAB.NS', 'CIPLA.NS', 'APOLLOHOSP.NS', 'TATACONSUM.NS', 
    'HEROMOTOCO.NS', 'DRREDDY.NS', 'BAJAJ-AUTO.NS', 'YESBANK.NS', 'IDEA.NS', 'UPL.NS', 
    'BANDHANBNK.NS', 'DLF.NS', 'VBL.NS', 'PIDILITIND.NS', 'SIEMENS.NS', 'HAL.NS', 'BEL.NS', 
    'ABB.NS', 'GAIL.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'TRENT.NS', 'GODREJCP.NS', 
    'DABUR.NS', 'CHOLAFIN.NS', 'TVSMOTOR.NS', 'POLYCAB.NS', 'HAVELLS.NS', 'LTIM.NS', 
    'SRF.NS', 'COLPAL.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'BHEL.NS', 'RECLTD.NS', 'PFC.NS', 
    'IRFC.NS', 'RVNL.NS', 'MAZDOCK.NS', 'COCHINSHIP.NS', 'IRCTC.NS', 'CONCOR.NS', 
    'TATACOMM.NS', 'LUPIN.NS', 'AUBANK.NS', 'IDFCFIRSTB.NS', 'FEDERALBNK.NS', 'RBLBANK.NS', 
    'ABCAPITAL.NS', 'MFSL.NS', 'LICHSGFIN.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS', 
    'ZENSARTECH.NS', 'KPITTECH.NS', 'TATAELXSI.NS', 'OIL.NS', 'HINDPETRO.NS', 'IOC.NS', 
    'MRPL.NS', 'CHENNPETRO.NS', 'PETRONET.NS', 'TATAPOWER.NS', 'NHPC.NS', 'SJVN.NS', 
    'TORNTPOWER.NS', 'CESC.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'NMDC.NS', 'NATIONALUM.NS', 
    'VEDL.NS', 'ASHOKLEY.NS', 'ESCORTS.NS', 'BALKRISIND.NS', 'MRF.NS', 'APOLLOTYRE.NS', 
    'TATACHEM.NS', 'DEEPAKNTR.NS', 'AARTIIND.NS', 'PIIND.NS', 'COROMANDEL.NS', 'GNFC.NS', 
    'JUBLFOOD.NS', 'PAGEIND.NS', 'METROBRAND.NS', 'BATAINDIA.NS', 'RELAXO.NS', 'ABFRL.NS', 
    'NYKAA.NS', 'PAYTM.NS', 'DELHIVERY.NS', 'INDIAMART.NS', 'POLICYBZR.NS', 'CARBORUNIV.NS', 
    'CUMMINSIND.NS', 'SKFINDIA.NS', 'TIMKEN.NS', 'VOLTAS.NS', 'BLUESTARCO.NS', 'DIXON.NS', 
    'AMBER.NS', 'HINDZINC.NS', 'GLENMARK.NS', 'ALKEM.NS', 'ASTRAL.NS', 'OBEROIRLTY.NS', 
    'PHOENIXLTD.NS', 'GODREJPROP.NS', 'BRIGADE.NS', 'PRESTIGE.NS', 'UNIONBANK.NS', 
    'MAHABANK.NS', 'J&KBANK.NS', 'KARURVYSYA.NS', 'SOUTHBANK.NS', 'POONAWALLA.NS', 
    'LTF.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'ANGELONE.NS', 'MCX.NS', 'BSE.NS', 'CDSL.NS', 
    'HUDCO.NS', 'NBCC.NS', 'KEC.NS', 'KPIL.NS', 'IRB.NS', 'RAYMOND.NS', 'KAYNES.NS', 
    'SYNGENE.NS', 'LAURUSLABS.NS', 'JBCHEPHARM.NS', 'ZYDUSLIFE.NS', 'NATCOPHARM.NS', 
    'NAM-INDIA.NS', 'HDFCAMC.NS', 'RADICO.NS', 'TIINDIA.NS', 'KEI.NS', 'EXIDEIND.NS'
]

FEATURES = ["Daily_Return", "RSI", "Volatility", "SMA20_Ratio", "SMA50_Ratio", "MACD_Ratio"]

def generate_base_model():
    all_data = []
    total_stocks = len(STOCKS)
    
    for index, stock in enumerate(STOCKS, 1):
        try:
            print(f"[{index}/{total_stocks}] Fetching & Processing: {stock}...", end="\r")
            
            df = yf.download(stock, period="10y", progress=False)
            if len(df) < 500:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            
            df = calculate_indicators(df)
            
            if df is not None:
                df['Target_Weekly'] = (df['Close'].shift(-5) > df['Close'] * 1.005).astype(int)
                df['Target_Monthly'] = (df['Close'].shift(-21) > df['Close'] * 1.005).astype(int)
                df['Target_Yearly'] = (df['Close'].shift(-252) > df['Close'] * 1.005).astype(int)
                all_data.append(df.dropna())
        except Exception:
            print(f"\nError processing {stock}. Skipping...")
            continue

    if not all_data:
        print("\nError: No valid data collected!")
        return

    print(f"\n\nData Collection Complete. Merging Datasets...")
    final_df = pd.concat(all_data)
    X = final_df[FEATURES]
    
    output_dir = "models/saved_models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    periods = {'weekly': 'Target_Weekly', 'monthly': 'Target_Monthly', 'yearly': 'Target_Yearly'}
    
    print(f"Training Multi-Horizon Gradient Boosting Models...")
    for name, col in periods.items():
        print(f"   - Building {name.capitalize()} Model...", end="\r")
        model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        )
        model.fit(X, final_df[col])
        joblib.dump(model, os.path.join(output_dir, f"model_{name}.pkl"))
        print(f"   {name.capitalize()} Model Saved.               ")

    print(f"\n{'='*50}")
    print("All Models Ready!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    generate_base_model()