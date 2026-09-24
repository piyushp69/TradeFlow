import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP_COLOR = "#00d68f"
DOWN_COLOR = "#ff4d5e"
GRID = "#232733"


def plot_price_action(df, days=None, chart_type="Candlestick"):
    """Price with SMA 50/200 and Bollinger bands, plus volume, RSI and MACD panels."""
    df = df.copy()
    close = df["Close"]
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    df["BB_Upper"], df["BB_Lower"] = sma20 + 2 * std20, sma20 - 2 * std20
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["MACD"], df["Signal"] = macd, macd.ewm(span=9, adjust=False).mean()
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = -diff.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    if days:
        df = df.tail(days)
    x = df["Date"]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.58, 0.12, 0.15, 0.15])

    fig.add_trace(go.Scatter(x=x, y=df["BB_Upper"], line=dict(width=0), hoverinfo="skip", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df["BB_Lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(125, 211, 252, 0.08)", name="Bollinger Band", hoverinfo="skip"), 1, 1)

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                     name="Price", increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR), 1, 1)
    else:
        fig.add_trace(go.Scatter(x=x, y=df["Close"], name="Close", line=dict(color="#7dd3fc", width=2)), 1, 1)

    fig.add_trace(go.Scatter(x=x, y=df["SMA50"], name="SMA 50", line=dict(color="#fbbf24", width=1.4)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=df["SMA200"], name="SMA 200", line=dict(color="#c084fc", width=1.4)), 1, 1)

    vol_colors = [UP_COLOR if c >= o else DOWN_COLOR for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=x, y=df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.6,
                         showlegend=False), 2, 1)

    fig.add_trace(go.Scatter(x=x, y=df["RSI"], name="RSI", line=dict(color="#7dd3fc", width=1.3),
                             showlegend=False), 3, 1)
    for level, color in ((70, DOWN_COLOR), (30, UP_COLOR)):
        fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"), row=3, col=1)

    hist = df["MACD"] - df["Signal"]
    fig.add_trace(go.Bar(x=x, y=hist, name="MACD Hist", showlegend=False,
                         marker_color=[UP_COLOR if h >= 0 else DOWN_COLOR for h in hist]), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD"], name="MACD", line=dict(color="#fbbf24", width=1.2),
                             showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=x, y=df["Signal"], name="Signal", line=dict(color="#c084fc", width=1.2),
                             showlegend=False), 4, 1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        height=780,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        bargap=0,
    )
    fig.update_xaxes(showgrid=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    for row, title in ((1, "Price (₹)"), (2, "Volume"), (3, "RSI"), (4, "MACD")):
        fig.update_yaxes(title_text=title, title_font=dict(size=11, color="#8b93a3"), row=row, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def _empty(message, height=260):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color="#8b93a3", size=13))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _base_layout(fig, height, title=None, showlegend=False):
    fig.update_layout(template="plotly_dark", height=height, title=title, showlegend=showlegend,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=40 if title else 10, b=0), hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def plot_relative_strength(frame, ticker, days=252):
    """Stock return vs NIFTY and vs its sector, taken from the model's own market features."""
    cols = [c for c in ("stock_nifty_relative_strength", "stock_sector_relative_strength") if c in frame]
    if not cols:
        return _empty("No market-context data available")
    tail = frame.tail(days)
    labels = {"stock_nifty_relative_strength": f"{ticker} vs NIFTY 50",
              "stock_sector_relative_strength": f"{ticker} vs sector"}
    colors = {"stock_nifty_relative_strength": "#7dd3fc", "stock_sector_relative_strength": "#fbbf24"}
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=tail["Date"], y=tail[col] * 100, name=labels[col],
                                 line=dict(color=colors[col], width=2)))
    fig.add_hline(y=0, line=dict(color="#4b5563", width=1, dash="dot"))
    fig.update_yaxes(title_text="20-day relative return (%)", title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, 300, showlegend=True)


def plot_vix_and_fx(frame, days=252):
    """India VIX level with the USD/INR 20-day move on a second axis."""
    if "india_vix" not in frame:
        return _empty("No India VIX data available")
    tail = frame.tail(days)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["india_vix"], name="India VIX",
                             line=dict(color="#ff4d5e", width=2)), secondary_y=False)
    if "usdinr_return_20" in tail:
        fig.add_trace(go.Scatter(x=tail["Date"], y=tail["usdinr_return_20"] * 100, name="USD/INR 20d %",
                                 line=dict(color="#00d68f", width=1.6)), secondary_y=True)
    fig.update_yaxes(title_text="VIX", secondary_y=False, gridcolor=GRID, title_font=dict(size=11, color="#8b93a3"))
    fig.update_yaxes(title_text="USD/INR 20d %", secondary_y=True, showgrid=False,
                     title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, 300, showlegend=True)


def plot_timeframe_rsi(snapshot):
    """RSI across timeframes with the 30/70 bands."""
    rows = [(tf, vals.get("RSI")) for tf, vals in snapshot.items() if vals.get("RSI") is not None]
    if not rows:
        return _empty("No timeframe data available")
    values = [r[1] for r in rows]
    colors = [DOWN_COLOR if v > 70 else UP_COLOR if v < 30 else "#7dd3fc" for v in values]
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=values, marker_color=colors,
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    for level, color in ((70, DOWN_COLOR), (30, UP_COLOR)):
        fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"))
    fig.update_yaxes(range=[0, 100], title_text="RSI", title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, 280)


def plot_timeframe_volatility(snapshot):
    """Volatility per bar by timeframe (not annualised: the bars cover different spans)."""
    rows = [(tf, vals.get("volatility")) for tf, vals in snapshot.items() if vals.get("volatility") is not None]
    if not rows:
        return _empty("No timeframe data available")
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] * 100 for r in rows], marker_color="#c084fc",
                           text=[f"{r[1] * 100:.2f}%" for r in rows], textposition="outside"))
    fig.update_yaxes(title_text="Std-dev per bar (%)", title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, 280)


def plot_importance(report, value_column, top_n=15, color="#7dd3fc", title=None):
    """Horizontal bars for one importance column of the feature-selection report."""
    if report is None or value_column not in report:
        return _empty("Run trainer.py to generate feature-importance reports")
    data = report.dropna(subset=[value_column]).nlargest(top_n, value_column).iloc[::-1]
    if data.empty:
        return _empty("No values for this method")
    fig = go.Figure(go.Bar(x=data[value_column], y=data["feature"], orientation="h", marker_color=color))
    fig.update_layout(yaxis=dict(automargin=True))
    return _base_layout(fig, max(280, 26 * len(data)), title)


def plot_shap_beeswarm(sample, top_n=12):
    """SHAP value per row, coloured by how high that feature's value was."""
    if not sample:
        return _empty("Run trainer.py to generate SHAP reports")
    shap_values, values = sample["shap"], sample["values"]
    order = shap_values.abs().mean().sort_values(ascending=False).head(top_n).index[::-1]
    fig = go.Figure()
    for i, feature in enumerate(order):
        ranks = values[feature].rank(pct=True)
        last = i == len(order) - 1
        fig.add_trace(go.Scatter(
            x=shap_values[feature], y=np.random.default_rng(i).normal(i, 0.09, len(shap_values)),
            mode="markers", name=feature, showlegend=False,
            marker=dict(size=4, opacity=0.55, color=ranks, colorscale="RdYlBu_r", cmin=0, cmax=1,
                        colorbar=dict(title="feature<br>value", thickness=10, tickvals=[0, 1],
                                      ticktext=["low", "high"]) if last else None),
            hovertemplate=f"{feature}<br>SHAP %{{x:.4f}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#4b5563", width=1))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=list(order), automargin=True)
    fig.update_xaxes(title_text="SHAP value (impact on the model log-odds)",
                     title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, max(320, 30 * len(order)))


def plot_shap_explanation(explanation):
    """Per-prediction SHAP contributions for the ticker on screen."""
    if not explanation:
        return _empty("Per-prediction explanations require a LightGBM model")
    data = sorted(explanation, key=lambda e: e["shap_value"])
    colors = [UP_COLOR if e["shap_value"] > 0 else DOWN_COLOR for e in data]
    labels = [f"{e['feature']} = {e['value']:.3g}" for e in data]
    fig = go.Figure(go.Bar(x=[e["shap_value"] for e in data], y=labels, orientation="h", marker_color=colors))
    fig.add_vline(x=0, line=dict(color="#4b5563", width=1))
    fig.update_layout(yaxis=dict(automargin=True))
    fig.update_xaxes(title_text="pushes P(UP) down  <->  pushes P(UP) up",
                     title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, max(280, 34 * len(data)))


def plot_correlation_heatmap(corr, features):
    if corr is None or corr.empty:
        return _empty("Run trainer.py to generate the correlation report")
    features = [f for f in features if f in corr.columns][:20]
    if not features:
        return _empty("No overlapping features to show")
    sub = corr.loc[features, features]
    fig = go.Figure(go.Heatmap(z=sub.values, x=features, y=features, zmin=-1, zmax=1, colorscale="RdBu",
                               reversescale=True, colorbar=dict(thickness=10)))
    fig.update_layout(xaxis=dict(tickangle=45, automargin=True), yaxis=dict(automargin=True))
    return _base_layout(fig, 520)


def plot_equity_curve(curve, baseline_curve=None):
    """Backtest equity against the equal-weight universe benchmark."""
    if curve is None or curve.empty:
        return _empty("Run trainer.py to generate backtest results")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(curve["Date"]), y=curve["equity"], name="Model strategy",
                             line=dict(color=UP_COLOR, width=2.2)))
    if "benchmark_equity" in curve:
        fig.add_trace(go.Scatter(x=pd.to_datetime(curve["Date"]), y=curve["benchmark_equity"],
                                 name="Equal-weight universe", line=dict(color="#7dd3fc", width=1.6, dash="dot")))
    if baseline_curve is not None and not baseline_curve.empty:
        fig.add_trace(go.Scatter(x=pd.to_datetime(baseline_curve["Date"]), y=baseline_curve["equity"],
                                 name="Existing model", line=dict(color="#fbbf24", width=1.4)))
    fig.update_yaxes(title_text="Portfolio value (INR)", title_font=dict(size=11, color="#8b93a3"))
    return _base_layout(fig, 360, showlegend=True)
