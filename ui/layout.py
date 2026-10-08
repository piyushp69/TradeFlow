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
    "BHARTIARTL.NS", "ITC.NS", "LT.NS", "TATASTEEL.NS",
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
        "XGBoost models trained on 180 NSE stocks with scale-free technical, trend and market-context "
        "features. Each horizon is validated on purged walk-forward folds and scored once on a held-out, "
        "most recent test period. Horizons without a measurable out-of-sample edge show no call."
    )
    st.sidebar.warning("Not financial advice. TradeFlow is a research project: any edge it has is small and "
                       "uncertain, and past patterns may not repeat.", icon="⚠️")
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
        ("🤖 3 horizons", "Weekly, monthly and yearly forecasts, each scored out of sample on purged "
                         "walk-forward folds and a held-out test period."),
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


NO_EDGE_COLOR = "#6b7384"
HORIZON_LABELS = (("weekly", "Weekly"), ("monthly", "Monthly"), ("yearly", "Yearly"))


def _num(value, fmt):
    return "—" if value is None or pd.isna(value) else format(value, fmt)


MIN_INDEPENDENT_PERIODS = 10


def _tstat_text(test, horizon_days):
    """The IC t-stat, unless the period holds too few non-overlapping horizons for it to mean anything."""
    periods = (test.get("days") or 0) / max(horizon_days or 1, 1)
    if periods < MIN_INDEPENDENT_PERIODS:
        return f"t-stat n/a: only {periods:.1f} non-overlapping periods"
    return f"t {_num(test.get('rank_ic_tstat'), '.1f')}"


def _test_line(test, horizon_days=None):
    period = " → ".join(test.get("period") or []) or "n/a"
    return (f"Test period {period}: rank IC {_num(test.get('rank_ic'), '+.3f')} "
            f"({_tstat_text(test, horizon_days)}), AUC {_num(test.get('auc_rel'), '.3f')}")


def _muted_card(label, horizon_text, title, text, meta):
    _html(f"""
        <div class="pred-card" style="--accent:{NO_EDGE_COLOR}">
            <div class="card-head">
                <p class="card-label">{label}</p><span class="card-horizon">{horizon_text}</span>
            </div>
            <p class="card-trend" style="font-size:24px">○ {title}</p>
            <p class="card-conf">{text}</p>
            <p class="card-meta">{meta}</p>
        </div>
    """)


def _backtest_line(evaluation):
    result = (evaluation.get("backtest") or {}).get("model") or {}
    metrics, benchmark = result.get("metrics") or {}, result.get("benchmark") or {}
    if not metrics:
        return ""
    return (f"<br>Test-period backtest, top 20% after costs: {_fmt(metrics.get('cumulative_return'))} vs "
            f"{_fmt(benchmark.get('cumulative_return'))} for the equal-weight universe")


def forecast_card(label, horizon_text, res):
    evaluation = res.get("evaluation") or {}
    test = evaluation.get("final_test") or {}
    edge = evaluation.get("edge") or {}
    if res["direction"] == "NO_EDGE":
        reasons = "; ".join(edge.get("reasons") or []) or "no out-of-sample evaluation found"
        _muted_card(label, horizon_text, "NO RELIABLE EDGE",
                    "Out-of-sample tests found no consistent signal at this horizon, so TradeFlow makes no call.",
                    f"Why: {html.escape(reasons)}.<br>{_test_line(test, evaluation.get('horizon_days'))}")
        return
    if res["direction"] == "NO_RANK":
        _muted_card(label, horizon_text, "RANK UNAVAILABLE",
                    "This horizon's calls rank the stock against the whole universe on the same session, and "
                    "that data could not be loaded right now. Try again later.",
                    html.escape(res.get("rank_basis") or ""))
        return

    accent, icon = DIRECTION_STYLE[res["direction"]]
    rank = res.get("percentile") or 0
    outcome = "beat the median stock" if res.get("target") == "relative" else "rise more than 0.5%"
    _html(f"""
        <div class="pred-card" style="--accent:{accent}">
            <div class="card-head">
                <p class="card-label">{label}</p><span class="card-horizon">{horizon_text}</span>
            </div>
            <p class="card-trend" style="font-size:26px">{icon} {res['label']}</p>
            <p class="card-conf">Ranks above {rank:.0f}% of the universe ({html.escape(res.get('rank_basis') or '')})
               · model P({outcome}) {res['probability']:.1f}%</p>
            <div class="prob-track">
                <div class="prob-fill" style="width:{rank:.1f}%"></div>
                <div class="prob-mark" style="left:40%"></div>
                <div class="prob-mark" style="left:60%"></div>
            </div>
            <div class="prob-legend"><span>rank in the universe, same session</span><span>markers = neutral zone</span></div>
            <p class="card-meta">{_test_line(test, evaluation.get('horizon_days'))} · top vs bottom quintile beat the median
               {_num(test.get('top_hit_rel'), '.1%')} vs {_num(test.get('bottom_hit_rel'), '.1%')}{_backtest_line(evaluation)}</p>
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
            "cannot be computed across the 10-year training history. They are shown here as live context "
            "only; the models use daily features.", icon="ℹ️")


# ---------------------------------------------------------------- feature analysis

def feature_analysis_section(horizon):
    report = reports.feature_importance(horizon)
    if report is None:
        st.info("No feature reports yet. Run `python trainer.py` to generate them.")
        return

    used = report[report["used"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate features", len(report))
    c2.metric("Used by the model", len(used))
    c3.metric("Pruned", int((~report["used"]).sum()))

    tabs = st.tabs(["SHAP", "Gain importance", "Single-feature IC", "Correlation", "Full table"])
    with tabs[0]:
        left, right = st.columns([1, 1])
        left.plotly_chart(visuals.plot_importance(used, "shap_importance", color="#c084fc",
                                                  title="Mean |SHAP|"), width="stretch")
        right.plotly_chart(visuals.plot_shap_beeswarm(reports.shap_sample(horizon)), width="stretch")
        st.caption("XGBoost TreeSHAP of the deployed model on a sample of its own training rows: how the model "
                   "uses each feature, not evidence that the feature predicts returns.")
    with tabs[1]:
        st.plotly_chart(visuals.plot_importance(used, "gain_share", color="#00d68f",
                                                title="Share of total split gain"), width="stretch")
    with tabs[2]:
        st.plotly_chart(visuals.plot_importance(report.assign(abs_ic=report["single_feature_rank_ic"].abs()),
                                                "abs_ic", color="#fbbf24",
                                                title="|Rank IC| of each feature alone (training period)"),
                        width="stretch")
        st.caption("Daily Spearman correlation between the feature and the forward return across stocks, "
                   "averaged over the training period. The sign is in the full table.")
    with tabs[3]:
        top = used.nlargest(20, "shap_importance")["feature"].tolist() if not used.empty else []
        st.plotly_chart(visuals.plot_correlation_heatmap(reports.correlation(horizon), top), width="stretch")
    with tabs[4]:
        st.dataframe(report, hide_index=True, width="stretch")
        st.caption("`kept_in_folds`: share of walk-forward folds in which pruning kept the feature "
                   "(1.0 when pruning is off).")


# ---------------------------------------------------------------- model performance

def _mean_std(summary, key, fmt=".3f"):
    stats = (summary or {}).get(key) or {}
    if stats.get("mean") is None:
        return "—"
    if stats.get("std") is None:
        return format(stats["mean"], fmt)
    return f"{format(stats['mean'], fmt)} ± {format(stats['std'], fmt.lstrip('+'))}"


def performance_overview(metrics):
    """Headline tables for every horizon, read from models/saved_models/metrics.json."""
    if not metrics:
        st.info("No evaluation results found. Run `python trainer.py` to train and evaluate the models.")
        return
    data, protocol = metrics.get("data") or {}, metrics.get("protocol") or {}
    st.caption(f"Trained {str(metrics.get('training_date', ''))[:10]} on {data.get('stocks')} NSE stocks, "
               f"daily data {data.get('data_start')} → {data.get('data_end')} ({data.get('source')}). "
               f"Validation: {protocol.get('validation')}; {protocol.get('purge')}. "
               f"Final test: {protocol.get('final_test')}.")

    test_rows, fold_rows, compare_rows = [], [], []
    for key, label in HORIZON_LABELS:
        entry = (metrics.get("horizons") or {}).get(key)
        if not entry:
            continue
        test, edge, summary = entry["final_test"], entry["edge"], entry["fold_summary"]
        test_rows.append({
            "Horizon": label, "Test period": " → ".join(test["period"]), "Rank IC": test["rank_ic"],
            "IC t-stat": (f"{test['rank_ic_tstat']:.2f}" if test.get("days", 0) / entry["horizon_days"]
                          >= MIN_INDEPENDENT_PERIODS else "n/a"),
            "AUC (beat median)": test["auc_rel"],
            "AUC (UP > 0.5%)": test["auc_abs"], "Top 20% beat median": test["top_hit_rel"],
            "Bottom 20% beat median": test["bottom_hit_rel"],
            "Edge": "reliable" if edge.get("reliable") else "none"})
        fold_rows.append({
            "Horizon": label, "Rank IC": _mean_std(summary, "rank_ic", "+.4f"),
            "Folds with IC > 0": f"{summary.get('positive_ic_folds')}/{summary.get('n_folds')}",
            "AUC (beat median)": _mean_std(summary, "auc_rel"), "AUC (UP > 0.5%)": _mean_std(summary, "auc_abs"),
            "Top / bottom 20% beat median": f"{summary['top_hit_rel']['mean']:.1%} / "
                                            f"{summary['bottom_hit_rel']['mean']:.1%}"})
        baseline = entry.get("baseline") or {}
        reported = baseline.get("originally_reported") or {}
        before = baseline.get("final_test") or {}
        compare_rows.append({
            "Horizon": label, "Old, as reported: AUC (UP)": reported.get("test_auc"),
            "Old, purged: AUC (UP)": before.get("auc_abs"), "Old, purged: rank IC": before.get("rank_ic"),
            "New: AUC (UP)": test["auc_abs"], "New: AUC (beat median)": test["auc_rel"],
            "New: rank IC": test["rank_ic"]})

    three = st.column_config.NumberColumn(format="%.3f")
    four = st.column_config.NumberColumn(format="%+.4f")
    pct = st.column_config.NumberColumn(format="percent")
    st.markdown("##### Final test period (held out, scored once)")
    st.dataframe(pd.DataFrame(test_rows), hide_index=True, width="stretch", column_config={
        "Rank IC": four, "AUC (beat median)": three,
        "AUC (UP > 0.5%)": three, "Top 20% beat median": pct, "Bottom 20% beat median": pct})
    st.caption(f"Edge rule, fixed before the test period was scored: {protocol.get('edge_rule')}. Rank IC is the "
               "daily Spearman correlation between the model score and the realised forward return across stocks. "
               "AUC 0.50 and IC 0 are a coin flip. The t-stat is left blank when the test period holds fewer than "
               f"{MIN_INDEPENDENT_PERIODS} non-overlapping horizons (the yearly test holds about one).")
    st.markdown("##### Walk-forward validation (5 purged folds, mean ± std)")
    st.dataframe(pd.DataFrame(fold_rows), hide_index=True, width="stretch")
    st.markdown("##### Before and after")
    st.dataframe(pd.DataFrame(compare_rows), hide_index=True, width="stretch", column_config={
        "Old, as reported: AUC (UP)": three, "Old, purged: AUC (UP)": three, "Old, purged: rank IC": four,
        "New: AUC (UP)": three, "New: AUC (beat median)": three, "New: rank IC": four})
    st.caption("\"Old, as reported\" is the previous pipeline's own test score (4 folds, NIFTY index in the test "
               "set, weekly features selected on the validation folds). \"Old, purged\" re-scores that same model "
               "configuration with the purged walk-forward split on the same test period. All on the final test "
               "period of each horizon.")


def model_performance_section(metrics, horizon_choice):
    entry = ((metrics or {}).get("horizons") or {}).get(horizon_choice)
    if not entry:
        st.info("No evaluation results for this horizon.")
        return
    edge, model = entry["edge"], entry["model"]

    st.markdown("##### Walk-forward folds")
    rows = [{"Fold": m["fold"], "Train": " → ".join(m["train_period"]), "Validate": " → ".join(m["period"]),
             "Gap (sessions)": m["gap_sessions"], "Rank IC": m["rank_ic"], "IC (next-open entry)": m["rank_ic_open"],
             "AUC (beat median)": m["auc_rel"], "AUC (UP)": m["auc_abs"], "Top 20% beat median": m["top_hit_rel"],
             "Bottom 20% beat median": m["bottom_hit_rel"]} for m in entry["folds"]]
    test = entry["final_test"]
    rows.append({"Fold": "final test", "Train": " → ".join(model["train_period"]), "Validate": " → ".join(test["period"]),
                 "Gap (sessions)": entry["split"]["test"]["gap_sessions"], "Rank IC": test["rank_ic"],
                 "IC (next-open entry)": test["rank_ic_open"], "AUC (beat median)": test["auc_rel"],
                 "AUC (UP)": test["auc_abs"], "Top 20% beat median": test["top_hit_rel"],
                 "Bottom 20% beat median": test["bottom_hit_rel"]})
    three, pct = st.column_config.NumberColumn(format="%.3f"), st.column_config.NumberColumn(format="percent")
    four = st.column_config.NumberColumn(format="%+.4f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Rank IC": four, "IC (next-open entry)": four, "AUC (beat median)": three, "AUC (UP)": three,
        "Top 20% beat median": pct, "Bottom 20% beat median": pct})
    st.caption("Each fold trains only on rows whose labels end before the fold starts, with a gap of at least "
               "the horizon. \"Next-open entry\" measures the same forecast against the return from the next "
               "session's open, which is what a trader acting on the close could capture.")

    st.markdown("##### Edge check")
    checks = edge.get("checks") or {}
    names = {"folds": "Positive rank IC in most walk-forward folds", "test": "Positive rank IC on the final test",
             "significance": "Out-of-sample IC statistically significant"}
    for key, text in names.items():
        st.markdown(f"{'✅' if checks.get(key) else '❌'} {text}")
    st.caption(f"Out-of-sample IC {edge.get('oos_ic_mean', 0):+.4f} over {edge.get('oos_days')} days, Newey-West "
               f"t-stat {_num(edge.get('oos_ic_tstat'), '.2f')}. Verdict: "
               f"{'reliable edge' if edge.get('reliable') else 'no reliable edge, so the app makes no call'}.")

    st.markdown("##### Model")
    st.markdown(f"XGBoost {model.get('version')} · target: {entry['target_description']} · "
                f"{model['n_features']} features ({model.get('candidate_features')} candidates) · "
                f"{model['n_trees']} trees · trained on {model['train_rows']:,} rows, "
                f"{' → '.join(model['train_period'])} · experiment step {model.get('experiment_step')}")
    with st.expander("Hyper-parameters and features"):
        st.json({"params": model["params"], "clear_move_k": model.get("clear_move_k"),
                 "cross_sectional": model.get("cross_sectional"), "early_stopping": model.get("early_stopping"),
                 "prune": model.get("prune"), "features": model["features"]})

    log = reports.experiments(horizon_choice)
    if log is not None:
        st.markdown("##### Experiment log (walk-forward folds only)")
        show = [c for c in ["step", "change", "rank_ic_mean", "rank_ic_std", "ic_by_fold", "auc_rel_mean",
                            "auc_abs_mean", "vs_reference", "decision"] if c in log]
        st.dataframe(log[show], hide_index=True, width="stretch", column_config={
            "rank_ic_mean": four, "rank_ic_std": st.column_config.NumberColumn(format="%.4f"),
            "auc_rel_mean": three, "auc_abs_mean": three})
        st.caption("A change was kept only if mean rank IC improved, IC improved in at least 4 of 5 folds and mean "
                   "AUC (beat median) did not fall. The final test period played no part in these decisions.")

    backtest_section(entry, horizon_choice)

    audit = metrics.get("leakage_audit") or {}
    if audit:
        failures = audit.get("look_ahead_failures") or []
        alignment = audit.get("intraday_alignment_problems") or {}
        if failures or alignment:
            st.error(f"Leakage audit found problems: {failures} {alignment}")
        else:
            st.success(f"Leakage audit: {audit.get('features_checked')} features rebuilt point-in-time on "
                       f"{', '.join(audit.get('stocks_checked', []))} with no look-ahead differences; every "
                       "split re-checked for label overlap before training.", icon="✅")


def backtest_section(entry, horizon):
    st.markdown("##### Backtest on the final test period")
    backtests = entry.get("backtest") or {}
    result = backtests.get("model") or {}
    metrics, benchmark, config = result.get("metrics") or {}, result.get("benchmark") or {}, result.get("config") or {}
    if not metrics:
        st.warning("Not enough non-overlapping periods in the test window to backtest this horizon.")
        return
    previous = (backtests.get("baseline") or {}).get("metrics") or {}
    cols = st.columns(5)
    _tile(cols[0], "Cumulative return", _fmt(metrics.get("cumulative_return")),
          f"benchmark {_fmt(benchmark.get('cumulative_return'))}")
    _tile(cols[1], "Sharpe ratio", _fmt(metrics.get("sharpe_ratio"), "raw"))
    _tile(cols[2], "Max drawdown", _fmt(metrics.get("max_drawdown")))
    _tile(cols[3], "Win rate", _fmt(metrics.get("win_rate"), "pct_abs"), f"{metrics.get('number_of_trades')} trades")
    _tile(cols[4], "Profit factor", _fmt(metrics.get("profit_factor"), "raw"),
          f"previous model {_fmt(previous.get('cumulative_return'))}" if previous else None)
    st.plotly_chart(visuals.plot_equity_curve(reports.equity_curve(horizon),
                                              reports.equity_curve(horizon, baseline=True)), width="stretch")
    note = " A single rebalance is an anecdote, not evidence." if metrics.get("periods", 0) < 6 else ""
    st.caption(
        f"Long-only, equal weight, top {config.get('top_quantile', 0.2):.0%} of the universe by model score, "
        f"rebalanced every {config.get('rebalance_days')} sessions, filled at the next session's open with "
        f"{config.get('transaction_cost_bps')} bps costs + {config.get('slippage_bps')} bps slippage, over "
        f"{metrics.get('periods')} periods.{note} The universe is today's stock list, so absolute returns carry "
        "survivorship bias; the equal-weight benchmark shares it, so the difference is the meaningful part. "
        "Shown for every horizon, including those without a reliable edge.")


def explanation_section(results):
    """Per-prediction SHAP explanation, only for horizons with a reliable edge (no call, nothing to explain)."""
    available = {h: r for h, r in results.items() if r.get("explanation")}
    if not available:
        return
    st.markdown("##### Why the model says this")
    st.caption("SHAP contributions of this stock minus those of the average universe stock on the same session. "
               "Market-wide inputs (NIFTY, India VIX, USD/INR) are the same for every stock and cancel out, so the "
               "bars show what sets this stock's rank apart.")
    tabs = st.tabs([h.capitalize() for h in available])
    for tab, (horizon, res) in zip(tabs, available.items()):
        with tab:
            tab.plotly_chart(visuals.plot_shap_explanation(res["explanation"]), width="stretch")
