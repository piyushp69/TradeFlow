import streamlit as st
import pandas as pd
from data.fetcher import get_stock_data
from features.indicators import calculate_indicators
from models.predictor import StockPredictor
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="TradeFlow | Advanced Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .main-title {
        font-size: 45px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f0f2f6;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #a1a7b3;
        font-size: 18px;
        margin-top: 0px;
        margin-bottom: 30px;
    }
    .pred-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #1a1c24;
        transition: transform 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .pred-card:hover {
        transform: translateY(-5px);
    }
    .up-card {
        border: 2px solid #00ff88;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.3);
    }
    .down-card {
        border: 2px solid #ff3131;
        box-shadow: 0 0 15px rgba(255, 49, 49, 0.3);
    }
    .card-label { font-size: 16px; text-transform: uppercase; letter-spacing: 1px; color: #a1a7b3; margin-bottom: 5px;}
    .card-trend { font-size: 36px; font-weight: 800; margin-top: 0; margin-bottom: 5px;}
    .card-conf { font-size: 14px; color: #888;}
    [data-testid="stMetricValue"] {
        font-size: 40px !important;
        color: #f0f2f6 !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #a1a7b3 !important;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">TradeFlow</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Insights & Multi-Horizon Trend Forecasting</p>', unsafe_allow_html=True)

st.sidebar.markdown("## Control Panel")
ticker = st.sidebar.text_input("NSE Ticker", "TCS.NS")
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Run Analytics")

placeholder = st.empty()

if predict_btn:
    with st.spinner(f"Processing Pipeline for {ticker}..."):
        placeholder.empty()

        data = get_stock_data(ticker)
        
        if data is not None and not data.empty:
            processed_data = calculate_indicators(data)
            current_price = data['Close'].iloc[-1]
            last_date = data['Date'].iloc[-1].strftime('%d %b, %Y')
            
            m1, m2 = st.columns([2, 1])
            with m1:
                 st.metric(label=f"Current Market Quote: {ticker} ({last_date})", value=f"₹{current_price:,.2f}")
            
            predictor = StockPredictor()
            results = predictor.predict_all(processed_data)
            
            if results:
                st.markdown("---")
                st.subheader("Forecast Summary")
                cols = st.columns(3)
                
                periods = [('Weekly', 'weekly'), ('Monthly', 'monthly'), ('Yearly', 'yearly')]
                
                for i, (label, key) in enumerate(periods):
                    if key in results:
                        direction, conf = results[key]
                        style_class = "up-card" if direction == "UP" else "down-card"
                        text_color = "#00ff88" if direction == "UP" else "#ff3131"
                        
                        with cols[i]:
                            st.markdown(f"""
                                <div class="pred-card {style_class}">
                                    <p class="card-label">{label}</p>
                                    <p class="card-trend" style="color:{text_color};">{direction}</p>
                                    <p class="card-conf">Confidence: {conf}%</p>
                                </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Technical Performance Chart")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'], 
                                         name="Close Price", line=dict(color='#17BECF', width=2.5)))

                rolling_std = processed_data['Close'].rolling(window=20).std()
                sma_20 = processed_data['Close'].rolling(window=20).mean()
                upper_band = sma_20 + (rolling_std * 2)
                lower_band = sma_20 - (rolling_std * 2)

                fig.add_trace(go.Scatter(x=processed_data['Date'], y=upper_band, line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=processed_data['Date'], y=lower_band, line=dict(width=0), fill='tonexty', fillcolor='rgba(23, 190, 207, 0.1)', name="Volatility Band"))

                fig.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(gridcolor='#2d303b'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=50, b=0),
                    height=550
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Model state unavailable. Please verify model artifacts.")
        else:
             st.error("Data acquisition failed. Check ticker symbol and connectivity.")
else:
    with placeholder.container():
        st.markdown("---")
        st.markdown("""
        ### Welcome to TradeFlow 🚀
        Please provide a ticker symbol in the sidebar to initiate comprehensive market analysis.
        
        **System Overview:**
        - Real-time data processing via localized pipelines.
        - Advanced technical feature engineering.
        - Machine Learning forecasting across multiple temporal horizons.
        """)
        st.info("Supported Ticker Format: [STOCK NAME].NS")