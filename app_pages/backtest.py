import pandas as pd
import streamlit as st

from stocksense.backtest import monthly_returns, run_backtest
from stocksense.models import MODEL_NAMES
from ui import charts, state
from ui.components import card_title, chart, empty_state, footer

st.markdown("### Backtest")
st.caption("Trade the walk-forward (out-of-sample) forecasts: decide at the close, earn the next "
           "session's return, pay costs on every position change.")

result = state.lab_result()
if result is None:
    empty_state(":material/account_balance:", "No out-of-sample forecasts yet",
                "The backtest trades the predictions produced in the Model Lab, so the strategy is only "
                "ever tested on data the model had not seen. Run an evaluation first.",
                page="app_pages/model_lab.py", label="Go to Model Lab")
    footer()
    st.stop()

cfg = state.config().backtest
candidates = [m for m in result.models if m != "naive"]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1], vertical_alignment="bottom")
    with c1:
        model = st.selectbox("Signal model", candidates, format_func=MODEL_NAMES.get, key="bt_model")
    with c2:
        style = st.segmented_control("Strategy", ["Long / flat", "Long / short"], required=True,
                                     default="Long / short" if cfg.allow_short else "Long / flat", key="bt_style")
    with c3:
        threshold = st.slider("Entry threshold (bps)", 0, 50, int(cfg.threshold * 1e4), key="bt_thr",
                              help="Only trade when |predicted return| exceeds this.")
    with c4:
        cost = st.slider("Cost per trade (bps)", 0, 30, int(cfg.cost_bps), key="bt_cost")

preds = result.predictions
bt = run_backtest(preds[model], preds["actual"], threshold / 1e4, cost, style == "Long / short")
s, bh = bt.strategy, bt.benchmark
name = MODEL_NAMES[model]

k = st.columns(5)
k[0].metric("Total return", f"{s['total_return']:+.1f}%", delta=f"{s['total_return'] - bh['total_return']:+.1f} pts vs buy & hold", border=True, height="stretch")
k[1].metric("Sharpe ratio", f"{s['sharpe']:.2f}", delta=f"{s['sharpe'] - bh['sharpe']:+.2f} vs buy & hold", border=True, height="stretch",
            help="Annualised mean / volatility of daily returns (risk-free rate = 0).")
k[2].metric("Max drawdown", f"{s['max_drawdown']:.1f}%", delta=f"{s['max_drawdown'] - bh['max_drawdown']:+.1f} pts",
            border=True, height="stretch", help="Largest peak-to-trough fall in equity.")
k[3].metric("Time in market", f"{s['exposure']:.0f}%", border=True, height="stretch")
k[4].metric("Trades", f"{s['trades']:,}", border=True, height="stretch", help="Number of position changes.")

daily = bt.daily
with st.container(border=True):
    card_title("Cumulative return", f"{name} strategy vs buy & hold over the same out-of-sample period",
               data=daily[["position", "strategy_equity", "benchmark_equity"]].round(4))
    chart(charts.equity_chart(daily, name), key="bt_equity")

a, b = st.columns([1, 1])
with a, st.container(border=True):
    card_title("Drawdown", "How far each curve sits below its previous peak")
    chart(charts.drawdown_chart(daily, name), key="bt_dd")
with b, st.container(border=True):
    card_title("Risk & return summary")
    rows = {"Total return": ("total_return", "%+.1f%%"), "CAGR": ("cagr", "%+.1f%%"),
            "Volatility (ann.)": ("volatility", "%.1f%%"), "Sharpe": ("sharpe", "%.2f"),
            "Sortino": ("sortino", "%.2f"), "Max drawdown": ("max_drawdown", "%.1f%%")}
    summary = pd.DataFrame({name: [fmt % s[k] for k, fmt in rows.values()],
                            "Buy & hold": [fmt % bh[k] for k, fmt in rows.values()]}, index=list(rows))
    st.dataframe(summary, height=250)

monthly = monthly_returns(daily["strategy_ret"])
with st.container(border=True):
    card_title("Monthly strategy returns", "Compounded % per calendar month",
               data=monthly.round(2))
    chart(charts.monthly_heatmap(monthly), key="bt_monthly")

st.info("**Read with care.** Close-to-close execution, no slippage or borrow costs, a single price path and "
        "a stock chosen with hindsight all flatter results. An edge that vanishes once costs rise to "
        "10–20 bps was never a real edge.", icon=":material/info:")
footer()
