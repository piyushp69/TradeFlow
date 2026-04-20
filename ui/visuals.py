import plotly.graph_objects as go
import streamlit as st

def plot_price_action(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        name="Close Price",
        line=dict(color='#00d4ff', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'].rolling(window=50).mean(),
        name="SMA 50",
        line=dict(color='#ff9f00', width=1.5, dash='dot')
    ))
    
    fig.update_layout(
        title=f"{ticker} Price Action",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0),
        height=450
    )
    return fig

def plot_indicator(df, column, name, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'].tail(100), 
        y=df[column].tail(100),
        name=name,
        line=dict(color=color)
    ))
    fig.update_layout(
        height=250,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        title=name
    )
    return fig