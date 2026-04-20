import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

def sidebar_content():
    st.sidebar.header("About TradeFlow")
    st.sidebar.info(
        "This system utilizes Gradient Boosting Machine Learning models "
        "to forecast stock trends based on historical price action and technical indicators.\n\n"
        "**Core Capabilities:**\n"
        "- Multi-Horizon Forecasting (Weekly, Monthly, Yearly)\n"
        "- Stock-Agnostic Normalized Feature Engineering\n"
        "- Interactive Technical Visualizations"
    )
    st.sidebar.warning("Disclaimer: Predictions are based on historical patterns. Past performance does not guarantee future results.")