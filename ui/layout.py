import html
import textwrap
import numpy as np
import pandas as pd
import streamlit as st

from models import reports
from ui import visuals

UP_COLOR = "#00d68f"
DOWN_COLOR = "#ff4d5e"
NEUTRAL_COLOR = "#fbbf24"
DIRECTION_STYLE = {"UP": (UP_COLOR, "▲"), "DOWN": (DOWN_COLOR, "▼"), "NEUTRAL": (NEUTRAL_COLOR, "◆")}

POPULAR_TICKERS = [
    "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "TATASTEEL.NS", "^NSEI",
]

RANGES = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": None}


def apply_custom_style():
    st.markdown(f"""
    <style>
        .block-container {{ padding-top: 2rem; max-width: 1400px; }}
        .main-title {{
            font-size: 42px; font-weight: 800; letter-spacing: -1px; margin-bottom: 0;
            background: linear-gradient(90deg, #7dd3fc, #00d68f);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .sub-title {{ color: #9aa3b2; font-size: 16px; margin-top: 0; margin-bottom: 1.5rem; }}

        .quote-box {{ display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; margin-bottom: 0.25rem; }}
        .quote-ticker {{ font-size: 15px; color: #9aa3b2; letter-spacing: 1px; text-transform: uppercase; }}
        .quote-price {{ font-size: 40px; font-weight: 750; color: #f0f2f6; }}
        .quote-change {{ font-size: 18px; font-weight: 600; }}
        .quote-date {{ color: #6b7384; font-size: 13px; }}

        .stat-card {{
            background: #161a23; border: 1px solid #262b36; border-radius: 12px;
            padding: 14px 16px; height: 100%;
        }}
        .stat-label {{ color: #8b93a3; font-size: 12px; text-transform: uppercase; letter-spacing: .8px; margin: 0; }}
        .stat-value {{ color: #f0f2f6; font-size: 20px; font-weight: 700; margin: 4px 0 0 0; }}

        .pred-card {{
            background: #161a23; border: 1px solid #262b36; border-radius: 16px;
            padding: 20px 22px; border-top: 4px solid var(--accent);
            transition: transform .2s, box-shadow .2s;
        }}
        .pred-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.35); }}
        .card-head {{ display: flex; justify-content: space-between; align-items: center; }}
        .card-label {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1.2px; color: #9aa3b2; margin: 0; }}
        .card-horizon {{ font-size: 12px; color: #6b7384; }}
        .card-trend {{ font-size: 34px; font-weight: 800; margin: 6px 0 2px 0; color: var(--accent); }}
        .card-conf {{ font-size: 14px; color: #c3c9d4; margin: 0 0 12px 0; }}
        .prob-track {{ position: relative; height: 8px; border-radius: 4px; background: #262b36; overflow: visible; }}
        .prob-fill {{ height: 100%; border-radius: 4px; background: var(--accent); }}
        .prob-mark {{ position: absolute; top: -4px; width: 2px; height: 16px; background: #e5e7eb; }}
        .prob-legend {{ display: flex; justify-content: space-between; font-size: 11px; color: #6b7384; margin-top: 6px; }}
        .card-meta {{ font-size: 12px; color: #8b93a3; margin: 12px 0 0 0; padding-top: 10px; border-top: 1px solid #262b36; }}

        .signal-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #262b36; }}
        .signal-name {{ color: #c3c9d4; }}
        .signal-val {{ font-weight: 600; }}
        .pill {{ padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-left: 8px; }}

        .stButton>button {{ width: 100%; border-radius: 8px; font-weight: 700; padding: 10px; }}
    </style>
    """, unsafe_allow_html=True)


def header():
    st.markdown('<p class="main-title">TradeFlow</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Automated insights & multi-horizon trend forecasting for NSE stocks</p>',
                unsafe_allow_html=True)


def sidebar_controls():
    st.sidebar.markdown("## Control Panel")
    ticker = st.sidebar.selectbox(
        "NSE Ticker", POPULAR_TICKERS, index=0, accept_new_options=True,
        help="Pick a popular ticker or type any Yahoo Finance symbol, e.g. WIPRO.NS",
    )
    run = st.sidebar.button("Run Analytics", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.caption(
        "Gradient-boosting models trained on ~180 NSE stocks with scale-free technical, trend and "
        "market-context features chosen per horizon. Each horizon is backtested on a held-out, most-recent "
        "time period."
    )
    st.sidebar.warning("Not financial advice. Predictions are based on historical patterns and are "
                       "only slightly better than chance.", icon="⚠️")
    return (ticker or "").strip().upper(), run


def _html(markup):
    # Dedent so Markdown doesn't treat indented HTML as a code block
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def welcome():
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    blocks = [
        ("📥 Live data", "10 years of daily prices from Yahoo Finance, plus NIFTY 50 market context."),
        ("🧮 Scale-free features", "Momentum, trend, oscillators, volatility, volume, market and sector context."),
        ("🤖 3 horizons", "Weekly, monthly and yearly direction, each with honest backtest metrics."),
    ]
    for col, (title, text) in zip((c1, c2, c3), blocks):
        col.markdown(f'<div class="stat-card"><p class="stat-value">{title}</p>'
                     f'<p class="card-conf" style="margin-top:8px">{text}</p></div>', unsafe_allow_html=True)
    st.info("Choose a ticker in the sidebar and press **Run Analytics**. Format: `SYMBOL.NS` (e.g. `INFY.NS`).")


def _pct(value, signed=True):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _period_return(close, days):
    return close.iloc[-1] / close.iloc[-days - 1] - 1 if len(close) > days else None


def quote_header(ticker, raw):
    close = raw["Close"]
    change = close.iloc[-1] / close.iloc[-2] - 1 if len(close) > 1 else 0
    color = UP_COLOR if change >= 0 else DOWN_COLOR
    arrow = "▲" if change >= 0 else "▼"
    _html(f"""
        <div class="quote-box">
            <span class="quote-ticker">{html.escape(ticker)}</span>
            <span class="quote-price">₹{close.iloc[-1]:,.2f}</span>
            <span class="quote-change" style="color:{color}">{arrow} {_pct(change)}</span>
            <span class="quote-date">as of {raw['Date'].iloc[-1]:%d %b %Y}</span>
        </div>
    """)


def stats_row(raw):
    close = raw["Close"]
    last_year = raw.tail(252)
    ann_vol = np.log(close).diff().tail(252).std() * np.sqrt(252)
    stats = [
        ("1M Return", _pct(_period_return(close, 21))),
        ("1Y Return", _pct(_period_return(close, 252))),
        ("52W High", f"₹{last_year['High'].max():,.2f}"),
        ("52W Low", f"₹{last_year['Low'].min():,.2f}"),
        ("Volatility (1Y)", _pct(ann_vol, signed=False)),
    ]
    for col, (label, value) in zip(st.columns(len(stats)), stats):
        col.markdown(f'<div class="stat-card"><p class="stat-label">{label}</p>'
                     f'<p class="stat-value">{value}</p></div>', unsafe_allow_html=True)


def forecast_card(label, horizon_text, res):
    accent, icon = DIRECTION_STYLE[res["direction"]]
    prob = res["up_probability"]
    rank = res.get("percentile") or 0
    metrics = res.get("test_metrics", {})
    _html(f"""
        <div class="pred-card" style="--accent:{accent}">
            <div class="card-head">
                <p class="card-label">{label}</p><span class="card-horizon">{horizon_text}</span>
            </div>
            <p class="card-trend">{icon} {res['direction']}</p>
            <p class="card-conf">P(UP) {prob:.1f}% vs universe median {res['threshold'] * 100:.1f}%</p>
            <div class="prob-track">
                <div class="prob-fill" style="width:{rank:.1f}%"></div>
                <div class="prob-mark" style="left:40%"></div>
                <div class="prob-mark" style="left:60%"></div>
            </div>
            <div class="prob-legend"><span>Ranks above {rank:.0f}% of NSE universe</span><span>markers = neutral zone</span></div>
            <p class="card-meta">Test AUC {metrics.get('roc_auc', 0):.3f} ·
               top-quintile hit rate {metrics.get('up_rate_top_quintile', 0):.1%}
               vs {metrics.get('up_rate_bottom_quintile', 0):.1%} bottom</p>
        </div>
    """)


def technical_signals(latest):
    def row(name, value, tag, positive):
        color = UP_COLOR if positive else DOWN_COLOR if positive is False else "#9aa3b2"
        return (f'<div class="signal-row"><span class="signal-name">{name}</span><span class="signal-val">{value}'
                f'<span class="pill" style="background:{color}22;color:{color}">{tag}</span></span></div>')

    rsi = latest["RSI"]
    rsi_tag, rsi_pos = ("Overbought", False) if rsi > 70 else ("Oversold", True) if rsi < 30 else ("Neutral", None)
    rows = [
        row("RSI (14)", f"{rsi:.1f}", rsi_tag, rsi_pos),
        row("Price vs 50-day SMA", _pct(latest["SMA50_Ratio"] - 1),
            "Above" if latest["SMA50_Ratio"] > 1 else "Below", latest["SMA50_Ratio"] > 1),
        row("Price vs 200-day SMA", _pct(latest["SMA200_Ratio"] - 1),
            "Uptrend" if latest["SMA200_Ratio"] > 1 else "Downtrend", latest["SMA200_Ratio"] > 1),
        row("50 / 200-day SMA", _pct(latest["SMA50_200_Ratio"] - 1),
            "Golden cross" if latest["SMA50_200_Ratio"] > 1 else "Death cross", latest["SMA50_200_Ratio"] > 1),
        row("MACD histogram", f"{latest['MACD_Hist_Ratio'] * 100:+.2f}%",
            "Bullish" if latest["MACD_Hist_Ratio"] > 0 else "Bearish", latest["MACD_Hist_Ratio"] > 0),
        row("Distance from 52W high", _pct(latest["High252_Dist"]), "", None),
    ]
    if not pd.isna(latest.get("Rel_Ret_63")):
        rows.append(row("3M return vs NIFTY 50", _pct(latest["Rel_Ret_63"]),
                        "Outperforming" if latest["Rel_Ret_63"] > 0 else "Underperforming", latest["Rel_Ret_63"] > 0))
    st.markdown("".join(rows), unsafe_allow_html=True)


def model_insights(results):
    rows = []
    for key, label in (("weekly", "Weekly"), ("monthly", "Monthly"), ("yearly", "Yearly")):
        if key not in results:
            continue
        m = results[key].get("test_metrics", {})
        rows.append({
            "Horizon": label,
            "ROC AUC": m.get("roc_auc"),
            "Balanced accuracy": m.get("balanced_accuracy"),
            "UP rate: top 20% ranked": m.get("up_rate_top_quintile"),
            "UP rate: bottom 20% ranked": m.get("up_rate_bottom_quintile"),
            "UP rate: all stocks": m.get("up_rate"),
            "Test rows": m.get("rows"),
            "Test period": " -> ".join(results[key].get("test_period") or []),
        })
    pct = st.column_config.NumberColumn(format="percent")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
            "Balanced accuracy": pct,
            "UP rate: top 20% ranked": pct,
            "UP rate: bottom 20% ranked": pct,
            "UP rate: all stocks": pct,
        },
    )
    st.caption(
        "Metrics come from the most recent period of data, which the models never saw during training or tuning. "
        "ROC AUC of 0.50 is a coin flip. Values in the 0.52–0.56 range are typical for daily-price-based models "
        "and mean the signal is weak.  \n**How to read a forecast:** the model ranks the stock against ~180 NSE stocks. "
        "It is UP when it ranks in the top 40%, DOWN in the bottom 40%, and NEUTRAL in between, so a forecast is relative: "
        "in a falling market, even top-ranked stocks can decline. \"UP\" means the price is more than 0.5% higher "
        "at the end of the horizon."
    )


# ---------------------------------------------------------------- market context

def _tile(col, label, value, hint=None, color=None):
    style = f' style="color:{color}"' if color else ""
    extra = f'<p class="card-conf" style="margin:6px 0 0 0;font-size:12px">{hint}</p>' if hint else ""
    col.markdown(f'<div class="stat-card"><p class="stat-label">{label}</p>'
                 f'<p class="stat-value"{style}>{value}</p>{extra}</div>', unsafe_allow_html=True)


def _trend_text(value):
    if value is None or pd.isna(value):
        return "—", None
    if value > 0:
        return "Uptrend", UP_COLOR
    if value < 0:
        return "Downtrend", DOWN_COLOR
    return "Mixed", NEUTRAL_COLOR


def _fmt(value, kind="pct"):
    if value is None or pd.isna(value):
        return "—"
    if kind == "pct":
        return f"{value * 100:+.2f}%"
    if kind == "pct_abs":
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def market_context_section(snapshot, meta, frame, ticker):
    if not snapshot:
        st.info("Market-context features are unavailable for this ticker.")
        return

    st.markdown("##### NIFTY 50")
    cols = st.columns(5)
    trend_text, trend_color = _trend_text(snapshot.get("nifty_trend_direction"))
    _tile(cols[0], "NIFTY trend", trend_text, "price vs 50 & 200-day averages", trend_color)
    _tile(cols[1], "NIFTY 1-day", _fmt(snapshot.get("nifty_return_1")))
    _tile(cols[2], "NIFTY 20-day", _fmt(snapshot.get("nifty_return_20")))
    _tile(cols[3], "NIFTY volatility", _fmt(snapshot.get("nifty_volatility_20"), "pct_abs"), "20-day, per day")
    _tile(cols[4], "NIFTY momentum", _fmt(snapshot.get("nifty_momentum"), "raw"), "20-day return / volatility")

    sector = meta.get("sector") or "unmapped"
    source = meta.get("sector_source") or "unavailable"
    label = "index" if str(source).startswith("index") else "peer composite"
    st.markdown(f"##### Sector — {sector.replace('_', ' ').title()} "
                f"<span class='card-horizon'>({label}: {str(source).split(':')[-1]})</span>", unsafe_allow_html=True)
    cols = st.columns(5)
    _tile(cols[0], "Sector 1-day", _fmt(snapshot.get("sector_return_1")))
    _tile(cols[1], "Sector 5-day", _fmt(snapshot.get("sector_return_5")))
    _tile(cols[2], "Sector 20-day", _fmt(snapshot.get("sector_return_20")))
    _tile(cols[3], "Sector volatility", _fmt(snapshot.get("sector_volatility"), "pct_abs"), "20-day, per day")
    rel = snapshot.get("stock_sector_relative_strength")
    _tile(cols[4], f"{ticker} vs sector", _fmt(rel), "20-day",
          UP_COLOR if (rel or 0) > 0 else DOWN_COLOR if rel is not None else None)

    st.markdown("##### Volatility & currency")
    cols = st.columns(5)
    vix, vix_change = snapshot.get("india_vix"), snapshot.get("india_vix_change")
    _tile(cols[0], "India VIX", f"{vix:.2f}" if vix else "—",
          f"{vix_change:+.2f} vs previous close" if vix_change is not None else None,
          DOWN_COLOR if (vix or 0) > 20 else UP_COLOR if vix else None)
    _tile(cols[1], "VIX 20-day vol", _fmt(snapshot.get("india_vix_volatility"), "pct_abs"))
    _tile(cols[2], "USD/INR 1-day", _fmt(snapshot.get("usdinr_return_1")), "lagged one session")
    _tile(cols[3], "USD/INR 20-day", _fmt(snapshot.get("usdinr_return_20")))
    nifty_rel = snapshot.get("stock_nifty_relative_strength")
    _tile(cols[4], f"{ticker} vs NIFTY", _fmt(nifty_rel), "20-day",
          UP_COLOR if (nifty_rel or 0) > 0 else DOWN_COLOR if nifty_rel is not None else None)

    st.markdown("")
    left, right = st.columns(2)
    left.plotly_chart(visuals.plot_relative_strength(frame, ticker), width="stretch")
    right.plotly_chart(visuals.plot_vix_and_fx(frame), width="stretch")
    st.caption("USD/INR is lagged by one session because its daily bar closes after the NSE close; "
               "NIFTY, sector and India VIX trade in the same session and are aligned to the same date.")


# ---------------------------------------------------------------- multi-timeframe

TIMEFRAME_LABELS = {"5m": "5 minute", "15m": "15 minute", "1h": "1 hour", "daily": "Daily"}


def timeframe_section(snapshot, meta, model_timeframes):
    rows = []
    for prefix, label in TIMEFRAME_LABELS.items():
        values = snapshot.get(prefix, {})
        trend_text, _ = _trend_text(values.get("trend_direction"))
        rows.append({
            "Timeframe": label,
            "RSI": values.get("RSI"),
            "Volatility per bar": values.get("volatility"),
            "Last bar return": values.get("return"),
            "Momentum (z)": values.get("momentum_z"),
            "Price vs fast EMA": values.get("price_vs_EMA"),
            "MACD histogram": values.get("MACD_hist_norm"),
            "Trend": trend_text,
            "Used by model": "yes" if prefix in model_timeframes else "no",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "RSI": st.column_config.NumberColumn(format="%.1f"),
        "Volatility per bar": st.column_config.NumberColumn(format="percent"),
        "Last bar return": st.column_config.NumberColumn(format="percent"),
        "Momentum (z)": st.column_config.NumberColumn(format="%.2f"),
        "Price vs fast EMA": st.column_config.NumberColumn(format="percent"),
        "MACD histogram": st.column_config.NumberColumn(format="percent"),
    })

    left, right = st.columns(2)
    left.plotly_chart(visuals.plot_timeframe_rsi(snapshot), width="stretch")
    right.plotly_chart(visuals.plot_timeframe_volatility(snapshot), width="stretch")

    last_bars = meta.get("intraday_last_bar") or {}
    if last_bars:
        st.caption("Last intraday bar used: " +
                   ", ".join(f"{TIMEFRAME_LABELS.get(k, k)} ending {pd.Timestamp(v):%d %b %H:%M}"
                             for k, v in sorted(last_bars.items())))
    st.info("Yahoo Finance serves only ~60 days of 5m/15m bars and ~2 years of 1h bars, so those timeframes "
            "cannot be computed across the 10-year training history. They are shown here as live context; "
            "the **Model Performance** tab reports a separate study of whether 1h features help.", icon="ℹ️")


# ---------------------------------------------------------------- feature analysis

def feature_analysis_section(horizon):
    report = reports.feature_selection(horizon)
    if report is None:
        st.info("No feature-analysis reports yet. Run `python trainer.py` to generate them.")
        return

    selected = report[report["selected"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate features", len(report))
    c2.metric("Selected", int(report["selected"].sum()))
    c3.metric("Removed", int((~report["selected"]).sum()))

    tabs = st.tabs(["SHAP", "Model importance", "Mutual information", "Permutation", "Correlation", "Full report"])
    with tabs[0]:
        left, right = st.columns([1, 1])
        left.plotly_chart(visuals.plot_importance(report, "shap_importance", color="#c084fc",
                                                  title="Mean |SHAP| (validation folds)"), width="stretch")
        right.plotly_chart(visuals.plot_shap_beeswarm(reports.shap_sample(horizon)), width="stretch")
        st.caption("SHAP values come from LightGBM's exact TreeSHAP on validation rows the model did not train on.")
    with tabs[1]:
        left, right = st.columns(2)
        left.plotly_chart(visuals.plot_importance(report, "model_importance_gain", color="#00d68f",
                                                  title="Gain importance"), width="stretch")
        right.plotly_chart(visuals.plot_importance(report, "model_importance_split", color="#7dd3fc",
                                                   title="Split importance"), width="stretch")
    with tabs[2]:
        st.plotly_chart(visuals.plot_importance(report, "mutual_information", color="#fbbf24",
                                                title="Mutual information with the target"), width="stretch")
    with tabs[3]:
        st.plotly_chart(visuals.plot_importance(report, "permutation_importance", color="#ff4d5e",
                                                title="Permutation importance (AUC drop, validation folds)"),
                        width="stretch")
        st.caption("Measured out-of-sample: each feature is shuffled in the validation block and the loss of "
                   "ROC AUC is recorded. Negative values mean the feature was not helping there.")
    with tabs[4]:
        top = selected.nlargest(20, "shap_importance")["feature"].tolist() if not selected.empty else []
        st.plotly_chart(visuals.plot_correlation_heatmap(reports.correlation(horizon), top), width="stretch")
        redundant = report[report["removal_reason"].astype(str).str.startswith("redundant")]
        if not redundant.empty:
            st.caption("Dropped as redundant: " +
                       "; ".join(f"{r.feature} ({r.removal_reason})" for r in redundant.itertuples()))
    with tabs[5]:
        st.dataframe(report, hide_index=True, width="stretch")
        st.caption("`selected` is decided by agreement between mutual information, gain, permutation and SHAP "
                   "rankings; no single method removes a feature on its own.")


# ---------------------------------------------------------------- model performance

def _metrics_table(results, key, caption):
    rows = []
    for horizon, label in (("weekly", "Weekly"), ("monthly", "Monthly"), ("yearly", "Yearly")):
        if horizon not in results:
            continue
        metrics = results[horizon].get(key) or {}
        if not metrics:
            continue
        rows.append({"Horizon": label, **{k.replace("_", " ").capitalize(): v for k, v in metrics.items()}})
    if not rows:
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(caption)


def model_performance_section(results, horizon_choice):
    st.markdown("##### Chosen model per horizon")
    rows = []
    for horizon, label in (("weekly", "Weekly"), ("monthly", "Monthly"), ("yearly", "Yearly")):
        if horizon not in results:
            continue
        res = results[horizon]
        rows.append({"Horizon": label, "Model": res.get("experiment"), "Type": res.get("model_type"),
                     "Features": res.get("n_features"), "Validation AUC": res.get("cv_auc"),
                     "Test AUC": (res.get("test_metrics") or {}).get("roc_auc"),
                     "Test period": " -> ".join(res.get("test_period") or [])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("##### Test metrics (held-out period, never used for tuning)")
    _metrics_table(results, "test_metrics",
                   "Directional accuracy, precision, recall and F1 use the same-day median split. ROC AUC of 0.50 "
                   "is a coin flip. MAE/R2 do not apply to a direction model, so the Brier score and the RMSE of "
                   "the probability are given instead.")

    st.markdown("##### Baseline vs enhanced experiments")
    table = reports.experiments(horizon_choice)
    if table is None:
        st.info("Run `python trainer.py` to generate the experiment comparison.")
    else:
        show = [c for c in ["experiment", "model", "n_features", "cv_auc", "test_roc_auc",
                            "test_directional_accuracy", "test_up_rate_top_quintile",
                            "test_up_rate_bottom_quintile", "test_brier_score"] if c in table]
        st.dataframe(table[show], hide_index=True, width="stretch")
        st.caption("The model is chosen by validation (`cv_auc`) only; the test column is reported afterwards.")

    study = reports.intraday_study(horizon_choice)
    if study is not None:
        st.markdown("##### 1-hour timeframe study")
        show = [c for c in ["variant", "n_features", "cv_auc", "test_roc_auc", "test_directional_accuracy"]
                if c in study]
        st.dataframe(study[show], hide_index=True, width="stretch")
        st.caption("Run only on the window where 1h bars exist (~2 years), so it is separate from the main "
                   "model choice.")

    st.markdown("##### Backtest")
    backtest_data = reports.backtest(horizon_choice)
    if not backtest_data:
        st.info("Run `python trainer.py` to generate backtest results.")
        return
    enhanced = backtest_data.get("enhanced", {})
    metrics, benchmark = enhanced.get("metrics", {}), enhanced.get("benchmark", {})
    baseline_metrics = backtest_data.get("baseline", {}).get("metrics", {})
    if not metrics:
        st.warning("Not enough non-overlapping periods in the test window to backtest this horizon.")
        return

    cols = st.columns(5)
    _tile(cols[0], "Cumulative return", _fmt(metrics.get("cumulative_return")),
          f"benchmark {_fmt(benchmark.get('cumulative_return'))}")
    _tile(cols[1], "Sharpe ratio", _fmt(metrics.get("sharpe_ratio"), "raw"))
    _tile(cols[2], "Max drawdown", _fmt(metrics.get("max_drawdown")))
    _tile(cols[3], "Win rate", _fmt(metrics.get("win_rate"), "pct_abs"),
          f"{metrics.get('number_of_trades')} trades")
    _tile(cols[4], "Profit factor", _fmt(metrics.get("profit_factor"), "raw"),
          f"existing model {_fmt(baseline_metrics.get('cumulative_return'))}")

    st.plotly_chart(visuals.plot_equity_curve(reports.equity_curve(horizon_choice),
                                              reports.equity_curve(horizon_choice, baseline=True)), width="stretch")
    config = enhanced.get("config", {})
    st.caption(
        f"Long-only, equal weight, top {config.get('top_quantile', 0.2):.0%} of the universe by P(UP), "
        f"rebalanced every {config.get('rebalance_days')} sessions, filled at the next session's open with "
        f"{config.get('transaction_cost_bps')} bps costs + {config.get('slippage_bps')} bps slippage. "
        "Periods count: " f"{metrics.get('periods')}. The universe is today's stock list, so results carry "
        "survivorship bias; the benchmark shares it, which makes the difference the meaningful part."
    )

    audit = reports.leakage_audit()
    if audit:
        failures = audit.get("look_ahead_failures") or []
        alignment = audit.get("intraday_alignment_problems") or {}
        if failures or alignment:
            st.error(f"Leakage audit found problems: {failures} {alignment}")
        else:
            st.success(f"Leakage audit: {audit.get('features_checked')} features rebuilt point-in-time on "
                       f"{', '.join(audit.get('stocks_checked', []))} with no look-ahead differences.", icon="✅")


def explanation_section(results):
    """Per-prediction SHAP explanation for each horizon (LightGBM models only)."""
    available = {h: r for h, r in results.items() if r.get("explanation")}
    if not available:
        return
    st.markdown("##### Why the model says this")
    tabs = st.tabs([h.capitalize() for h in available])
    for tab, (horizon, res) in zip(tabs, available.items()):
        with tab:
            tab.plotly_chart(visuals.plot_shap_explanation(res["explanation"]), width="stretch")
