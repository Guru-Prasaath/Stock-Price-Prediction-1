"""StockSense — Streamlit entry point.

Run with:  streamlit run app.py
"""

import streamlit as st

from ui import state
from ui.theme import CURRENCY_SYMBOLS, inject_css

st.set_page_config(page_title="StockSense", page_icon=":material/monitoring:", layout="wide")
inject_css()

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page("app_pages/explore.py", title="Explore", icon=":material/query_stats:"),
    st.Page("app_pages/model_lab.py", title="Model Lab", icon=":material/science:"),
    st.Page("app_pages/forecast.py", title="Forecast", icon=":material/timeline:"),
    st.Page("app_pages/backtest.py", title="Backtest", icon=":material/account_balance:"),
    st.Page("app_pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
]
nav = st.navigation(pages, position="top")

# ── Global scope bar: shared by every page ────────────────────────────
title_col, ticker_col, period_col = st.columns([3.2, 1.25, 1.1], vertical_alignment="bottom")
with ticker_col:
    st.selectbox(
        "Ticker", list(state.POPULAR), key="ticker", accept_new_options=True,
        format_func=lambda t: f"{t} · {state.POPULAR[t]}" if t in state.POPULAR else t.upper(),
        help="Pick a popular symbol or type any Yahoo Finance ticker (e.g. ADBE, BMW.DE, ^NSEI).",
    )
with period_col:
    st.segmented_control("History", list(state.PERIODS), key="period", default="5Y",
                         required=True, width="stretch")

ticker, years = state.selection()
with st.spinner(f"Loading {ticker}…"):
    ds = state.current_dataset()
    info = state.company(ticker)

prices = ds.prices
last, prev = float(prices["Close"].iloc[-1]), float(prices["Close"].iloc[-2])
change, change_pct = last - prev, (last / prev - 1) * 100
direction = "ss-up" if change >= 0 else "ss-down"
arrow = "▲" if change >= 0 else "▼"
meta = " · ".join(x for x in (ticker, info.get("exchange"), info.get("sector")) if x)

with title_col:
    st.markdown(
        f'<div class="ss-eyebrow">{meta}</div>'
        f'<h2 style="margin:0 0 0.25rem 0;padding:0">{info.get("name") or ticker}</h2>'
        f'<div><span class="ss-price">{CURRENCY_SYMBOLS.get(info.get("currency", "USD"), "")}{last:,.2f}</span>'
        f'<span class="ss-change {direction}">{arrow} {abs(change):,.2f} ({change_pct:+.2f}%)</span>'
        f'<span class="ss-muted">Close · {prices.index[-1]:%b %d, %Y}</span></div>',
        unsafe_allow_html=True,
    )
st.divider()

nav.run()
