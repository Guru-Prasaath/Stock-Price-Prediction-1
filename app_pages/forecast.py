import pandas as pd
import streamlit as st

from stocksense.features import FEATURE_DESCRIPTIONS
from ui import charts, state
from ui.components import card_title, chart, footer, money
from ui.theme import CURRENCY_SYMBOLS

ticker, years = state.selection()
ds = state.current_dataset()
info = state.company(ticker)
cur = info.get("currency", "USD")
sym = CURRENCY_SYMBOLS.get(cur, "")
cfg = state.config()

st.markdown("### Forecast")
st.caption("Direct multi-horizon XGBoost forecast with split-conformal prediction intervals — "
           "the bands show how wrong the model has actually been, not an assumed distribution.")

if years < state.MIN_YEARS_FOR_MODELS:
    st.warning("Choose at least 2 years of history to fit the forecast models.", icon=":material/warning:")
    st.stop()

c1, c2, _ = st.columns([1.4, 1.2, 2], vertical_alignment="bottom")
with c1:
    horizon = st.select_slider("Horizon (trading days)", [5, 10, 15, 20], value=cfg.forecast.horizon, key="fc_h")
with c2:
    context = st.segmented_control("History shown", ["3M", "6M", "1Y"], default="3M", required=True, key="fc_ctx")

with st.spinner(f"Fitting {horizon} horizon models…"):
    fc = state.forecast(ticker, years, horizon)
t = fc.table
first, final = t.iloc[0], t.iloc[-1]

k = st.columns(4)
k[0].metric(f"Last close · {fc.last_date:%b %d}", money(fc.last_close, cur), border=True, height="stretch")
k[1].metric(f"Next session · {first['date']:%b %d}", money(first["pred"], cur),
            delta=f"{(first['pred'] / fc.last_close - 1) * 100:+.2f}%", border=True, height="stretch")
k[2].metric(f"{horizon}-day forecast · {final['date']:%b %d}", money(final["pred"], cur),
            delta=f"{(final['pred'] / fc.last_close - 1) * 100:+.2f}%", border=True, height="stretch")
k[3].metric(f"80% range at day {horizon}", f"{money(final['lo_80'], cur, 0)} – {money(final['hi_80'], cur, 0)}",
            delta=f"±{(final['hi_80'] - final['lo_80']) / 2 / fc.last_close * 100:.1f}% width",
            delta_color="off", delta_arrow="off", border=True, height="stretch")

days = {"3M": 63, "6M": 126, "1Y": 252}[context]
history = ds.prices["Close"].iloc[-days:]
table = t[["date", "pred", "lo_80", "hi_80", "lo_95", "hi_95"]].copy()
table["change_pct"] = (table["pred"] / fc.last_close - 1) * 100

with st.container(border=True):
    card_title("Price forecast", "Median path with 80% and 95% calibrated intervals",
               data=table.set_index("date").round(2))
    chart(charts.fan_chart(history, t, fc.levels, sym), key="fc_fan")

a, b = st.columns([1.25, 1])
with a, st.container(border=True):
    card_title("Forecast table")
    def money_col(label: str):
        return st.column_config.NumberColumn(label, format=f"{sym}%.2f")

    st.dataframe(table, hide_index=True, height=min(38 + 35 * len(table), 420), column_config={
        "date": st.column_config.DateColumn("Date", format="ddd, MMM D"),
        "pred": money_col("Forecast"),
        "change_pct": st.column_config.NumberColumn("Change", format="%+.2f%%"),
        "lo_80": money_col("80% low"), "hi_80": money_col("80% high"),
        "lo_95": money_col("95% low"), "hi_95": money_col("95% high"),
    }, column_order=["date", "pred", "change_pct", "lo_80", "hi_80", "lo_95", "hi_95"])

with b, st.container(border=True):
    top = fc.contributions.reindex(fc.contributions.abs().sort_values(ascending=False).index).head(10)
    labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in top.index]
    card_title("What drives the next-session forecast",
               f"SHAP contributions in bps · base {fc.base_value * 1e4:+.1f} bps → forecast "
               f"{first['pred_ret'] * 1e4:+.1f} bps",
               data=pd.DataFrame({"feature": labels, "contribution (bps)": (top.values * 1e4).round(2)}).set_index("feature"))
    chart(charts.signed_bars(top * 1e4, labels, suffix=" bps", decimals=2, height=380), key="fc_shap")

st.warning("A forecast is not a recommendation. The Model Lab shows that next-day forecasts rarely beat a "
           "random walk with statistical significance; treat the median path as the model's best guess and "
           "the bands as the honest message.", icon=":material/info:")
footer()
