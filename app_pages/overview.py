import numpy as np
import pandas as pd
import streamlit as st

from ui import charts, state
from ui.components import big_money, card_title, chart, footer, money, pct
from ui.theme import CURRENCY_SYMBOLS

ticker, years = state.selection()
ds = state.current_dataset()
info = state.company(ticker)
cur = info.get("currency", "USD")
sym = CURRENCY_SYMBOLS.get(cur, "")
prices = ds.prices
close = prices["Close"]
ret = np.log(close).diff()


def period_return(days: int) -> float:
    return (close.iloc[-1] / close.iloc[-days - 1] - 1) * 100 if len(close) > days else float("nan")


# ── KPI row ──────────────────────────────────────────────────────────
year = close.iloc[-252:]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("1-month return", pct(period_return(21)), border=True, height="stretch",
          chart_data=close.iloc[-21:].round(2).tolist(), chart_type="area",
          help="Price change over the last 21 trading days.")
k2.metric("1-year return", pct(period_return(252)), border=True, height="stretch",
          chart_data=close.iloc[-252:].iloc[::5].round(2).tolist(), chart_type="area")
k3.metric("Volatility (1Y, ann.)", pct(ret.iloc[-252:].std() * np.sqrt(252) * 100, signed=False),
          border=True, height="stretch", help="Standard deviation of daily log returns × √252.")
k4.metric("From 52-week high", pct((close.iloc[-1] / year.max() - 1) * 100), border=True, height="stretch",
          delta=f"range {money(year.min(), cur, 0)} – {money(year.max(), cur, 0)}", delta_color="off",
          delta_arrow="off")
k5.metric("Market cap", big_money(info.get("market_cap"), cur), border=True, height="stretch")

# ── Price chart + side panel ─────────────────────────────────────────
main, side = st.columns([3, 1.15])
with main:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1, 1.6], vertical_alignment="center")
        with c1:
            st.markdown('<p class="ss-card-title">Price & volume</p>', unsafe_allow_html=True)
        with c2:
            kind = st.segmented_control("Chart type", ["Candles", "Line"], default="Candles",
                                        required=True, label_visibility="collapsed", key="ov_kind")
        with c3:
            window = st.segmented_control("Range", ["1M", "3M", "6M", "1Y", "All"], default="6M",
                                          required=True, label_visibility="collapsed", key="ov_range")
        overlays = st.pills("Overlays", ["SMA 20", "SMA 50", "SMA 200", "Bollinger"],
                            selection_mode="multi", default=["SMA 50"], label_visibility="collapsed",
                            key="ov_overlays")
        days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "All": len(prices)}[window]
        view = prices.iloc[-days:]
        chart(charts.price_chart(view, kind, overlays, sym), key="ov_price")

with side:
    with st.container(border=True):
        perf = pd.Series({
            "1 week": period_return(5), "1 month": period_return(21), "3 months": period_return(63),
            "6 months": period_return(126), "1 year": period_return(252),
            f"{years} years": (close.iloc[-1] / close.iloc[0] - 1) * 100,
        }).dropna()
        card_title("Performance", "Total price return",
                   data=perf.rename("Return %").round(2).to_frame())
        chart(charts.period_returns_chart(perf), key="ov_perf")

    with st.container(border=True):
        card_title("Profile")
        rows = {
            "Sector": info.get("sector") or "—",
            "Industry": info.get("industry") or "—",
            "Exchange": info.get("exchange") or "—",
            "Currency": cur,
            "History": f"{prices.index[0]:%b %Y} – {prices.index[-1]:%b %Y}",
            "Trading days": f"{len(prices):,}",
        }
        st.markdown("\n".join(f"- :gray[{k}] &nbsp; **{v}**" for k, v in rows.items()))
        if info.get("summary"):
            with st.expander("About the company"):
                st.write(info["summary"])

# ── Next steps ───────────────────────────────────────────────────────
with st.container(border=True):
    a, b, c = st.columns(3)
    with a:
        st.markdown("**1 · Understand the data**  \n:gray[Return distribution, volatility clustering, "
                    "stationarity and feature signal.]")
        st.page_link("app_pages/explore.py", label="Explore", icon=":material/query_stats:")
    with b:
        st.markdown("**2 · Compare models honestly**  \n:gray[Walk-forward evaluation of six models "
                    "against the random-walk benchmark.]")
        st.page_link("app_pages/model_lab.py", label="Model Lab", icon=":material/science:")
    with c:
        st.markdown("**3 · Forecast & backtest**  \n:gray[Calibrated multi-day forecast and a "
                    "cost-aware trading backtest.]")
        st.page_link("app_pages/forecast.py", label="Forecast", icon=":material/timeline:")

footer()
