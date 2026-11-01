import pandas as pd
import streamlit as st

from stocksense.features import FEATURE_DESCRIPTIONS
from stocksense.models import MODEL_CLASSES
from ui.components import card_title, footer

st.markdown("### Methodology")
st.caption("How StockSense turns raw prices into honest, testable forecasts.")

PIPELINE = """
digraph {
  rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.35; ranksep=0.45;
  node [shape=box style="rounded,filled" fillcolor="#1f1f1d" color="#383835" fontcolor="#f0efec"
        fontname="Inter" fontsize=11 margin="0.18,0.1"];
  edge [color="#6f6e69" arrowsize=0.6];
  data [label="Yahoo Finance\\nOHLCV + SPY + VIX"];
  feat [label="Features\\n23 causal signals"];
  wf   [label="Walk-forward\\nexpanding window"];
  models [label="6 models\\nRW · Drift · ARIMA\\nRidge · XGBoost · LSTM" fillcolor="#1c2a3d" color="#3987e5"];
  eval [label="Scoring\\nRMSE · R²oos · DM test"];
  bt   [label="Backtest\\ncost-aware"];
  fc   [label="Forecast\\nconformal intervals"];
  data -> feat -> wf -> models -> eval -> bt;
  models -> fc;
}
"""
with st.container(border=True):
    card_title("Pipeline")
    st.graphviz_chart(PIPELINE)

a, b = st.columns(2)
with a:
    with st.container(border=True):
        card_title("1 · Problem framing")
        st.markdown(
            "- **Target:** next-day log return $r_{t+1} = \\ln(C_{t+1}/C_t)$, not the price level. "
            "Prices are non-stationary (ADF test, *Explore* page); returns are not.\n"
            "- A return forecast converts back to a price as $\\hat C_{t+1} = C_t\\,e^{\\hat r_{t+1}}$.\n"
            "- **Benchmark:** the random walk $\\hat r = 0$. Every model must beat it to matter.")
    with st.container(border=True):
        card_title("2 · Leakage controls")
        st.markdown(
            "- Every feature on day *t* uses data up to the close of *t* only — verified by a test that "
            "truncates the future and checks past features are unchanged.\n"
            "- Scalers are fitted inside each training fold (the original project fitted on all data).\n"
            "- Market context is forward-filled, never back-filled.\n"
            "- A model-contract test perturbs future rows and asserts past predictions do not move.")
    with st.container(border=True):
        card_title("3 · Evaluation")
        st.markdown(
            "- **Walk-forward:** the first 60% of history trains fold 1; each later fold retrains on all "
            "data before its test block. All reported metrics are out-of-sample.\n"
            "- **Metrics:** price RMSE / MAPE (intuitive), return RMSE (ranking), directional accuracy, "
            "and Campbell–Thompson $R^2_{OOS}$ against the zero forecast.\n"
            "- **Significance:** Diebold–Mariano test (HLN-corrected) of squared errors vs the random walk; "
            "binomial test of directional accuracy vs the majority-class rate.")
with b:
    with st.container(border=True):
        card_title("4 · Backtest")
        st.markdown(
            "- Position decided at the close of *t* from the out-of-sample forecast; it earns the return "
            "of *t+1*.\n"
            "- Long/flat or long/short, optional entry threshold, transaction costs charged on turnover.\n"
            "- Reported against buy & hold over the identical period: return, CAGR, volatility, Sharpe, "
            "Sortino, max drawdown, exposure.")
    with st.container(border=True):
        card_title("5 · Forecast intervals")
        st.markdown(
            "- **Direct strategy:** one XGBoost model per horizon *h* predicts $\\ln(C_{t+h}/C_t)$.\n"
            "- **Split-conformal intervals:** each horizon's model is fitted on the first 80% of history; "
            "quantiles of its residuals on the last 20% set the band widths. The final model is refit on "
            "all data.\n"
            "- **Explainability:** exact TreeSHAP decomposes tomorrow's forecast into feature contributions.")
    with st.container(border=True):
        card_title("6 · Limitations")
        st.markdown(
            "- Daily bars only — no news, fundamentals, order-book or intraday data.\n"
            "- One price path per ticker; results vary by stock and period.\n"
            "- Backtests ignore slippage, borrow costs and taxes.\n"
            "- **Educational project — not investment advice.**")

c, d = st.columns(2)
with c, st.container(border=True):
    card_title("Models")
    st.dataframe(pd.DataFrame([{"Model": cls.name, "How it works": cls.description}
                               for cls in MODEL_CLASSES.values()]), hide_index=True)
with d, st.container(border=True):
    card_title("Features")
    st.dataframe(pd.DataFrame({"Feature": list(FEATURE_DESCRIPTIONS), "Meaning": list(FEATURE_DESCRIPTIONS.values())}),
                 hide_index=True, height=300)

with st.container(border=True):
    card_title("Reproduce from the command line")
    st.code("python -m scripts.train --ticker AAPL --track   # evaluate, backtest, forecast, save artifacts\n"
            "mlflow ui --backend-store-uri sqlite:///mlflow.db  # browse tracked experiments\n"
            "pytest                                             # leakage & correctness tests",
            language="bash")
footer()
