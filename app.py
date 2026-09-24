import streamlit as st

from models.inference import load_live_frame, market_snapshot, timeframe_snapshot
from models.predictor import StockPredictor
from ui import layout
from ui.visuals import plot_price_action

st.set_page_config(
    page_title="TradeFlow | Advanced Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_predictor():
    return StockPredictor()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ticker(symbol):
    """Prices, the full feature frame (same builder as training) and metadata for one ticker."""
    return load_live_frame(symbol)


layout.apply_custom_style()
layout.header()

ticker, run = layout.sidebar_controls()
if run and ticker:
    st.session_state["active_ticker"] = ticker

# Keep showing the last analysed ticker while the user interacts with charts and tabs
active = st.session_state.get("active_ticker")

if not active:
    layout.welcome()
    st.stop()

with st.spinner(f"Processing pipeline for {active}..."):
    raw, processed, meta = load_ticker(active)

if raw is None or raw.empty:
    st.error(f"Data acquisition failed for **{active}**. Check the ticker symbol (e.g. `INFY.NS`) and your connection.")
    st.stop()

layout.quote_header(active, raw)
layout.stats_row(raw)

if processed is None:
    st.warning("Not enough price history to forecast this ticker (at least ~1 year of trading data is required).")
    st.plotly_chart(plot_price_action(raw), width="stretch")
    st.stop()

predictor = load_predictor()
results = predictor.predict_all(processed, explain=True) or {}

st.markdown("### Forecast Summary")
if results:
    periods = [("Weekly", "weekly", "next 5 sessions"), ("Monthly", "monthly", "next 21 sessions"),
               ("Yearly", "yearly", "next 252 sessions")]
    for col, (label, key, horizon_text) in zip(st.columns(3), periods):
        if key in results:
            with col:
                layout.forecast_card(label, horizon_text, results[key])
else:
    st.error("Model state unavailable. Run `python trainer.py` to build the model artifacts.")

st.markdown("")
chart_tab, market_tab, timeframe_tab, signals_tab, features_tab, model_tab = st.tabs(
    ["📈 Technical Chart", "🌐 Market Context", "⏱ Multi-Timeframe", "🧭 Signals",
     "🔬 Feature Analysis", "🤖 Model Performance"])

with chart_tab:
    c1, c2 = st.columns([3, 1])
    with c1:
        range_label = st.segmented_control("Range", list(layout.RANGES), default="1Y", label_visibility="collapsed")
    with c2:
        chart_type = st.segmented_control("Chart", ["Candlestick", "Line"], default="Candlestick",
                                          label_visibility="collapsed")
    fig = plot_price_action(raw, layout.RANGES.get(range_label or "1Y"), chart_type or "Candlestick")
    st.plotly_chart(fig, width="stretch")

with market_tab:
    layout.market_context_section(market_snapshot(processed), meta, processed, active)

with timeframe_tab:
    model_timeframes = {tf for res in results.values() for group in (res.get("feature_groups") or [])
                        for tf in (("daily",) if group == "daily_mtf" else
                                   ("1h",) if group == "intraday_1h" else
                                   ("5m", "15m") if group == "intraday_short" else ())}
    layout.timeframe_section(timeframe_snapshot(processed), meta, model_timeframes)

with signals_tab:
    st.caption(f"Latest technical readings for {active}. The model uses these (and more) as inputs.")
    layout.technical_signals(processed.iloc[-1])
    layout.explanation_section(results)

with features_tab:
    horizon = st.segmented_control("Horizon", ["weekly", "monthly", "yearly"], default="weekly",
                                   key="feature_horizon", label_visibility="collapsed")
    layout.feature_analysis_section(horizon or "weekly")

with model_tab:
    if results:
        layout.model_insights(results)
        st.markdown("---")
        horizon = st.segmented_control("Horizon", ["weekly", "monthly", "yearly"], default="weekly",
                                       key="model_horizon", label_visibility="collapsed")
        layout.model_performance_section(results, horizon or "weekly")
    else:
        st.info("No trained models found.")
