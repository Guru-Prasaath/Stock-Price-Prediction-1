import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller

from stocksense.features import FEATURE_DESCRIPTIONS, feature_columns, labeled
from ui import charts, state
from ui.components import card_title, chart, footer
from ui.theme import ACCENT, BENCHMARK_COLOR

ticker, years = state.selection()
ds = state.current_dataset()
prices, frame = ds.prices, ds.frame
ret = np.log(prices["Close"]).diff().dropna()

st.markdown("### Exploratory analysis")
st.caption("What the data says before any model is trained — and why the modelling choices follow.")

tab_ret, tab_vol, tab_acf, tab_stat, tab_feat, tab_data = st.tabs(
    ["Returns", "Volatility", "Autocorrelation", "Stationarity", "Feature signal", "Data"])

# ── Returns distribution ─────────────────────────────────────────────
with tab_ret:
    skew, kurt = float(stats.skew(ret)), float(stats.kurtosis(ret))
    jb_p = float(stats.jarque_bera(ret).pvalue)
    z = (ret - ret.mean()) / ret.std()
    tail_obs = float((z.abs() > 3).mean() * 100)
    tail_norm = float(2 * stats.norm.sf(3) * 100)

    m = st.columns(5)
    m[0].metric("Mean return (ann.)", f"{ret.mean() * 252 * 100:+.1f}%", border=True, height="stretch")
    m[1].metric("Volatility (ann.)", f"{ret.std() * np.sqrt(252) * 100:.1f}%", border=True, height="stretch")
    m[2].metric("Skewness", f"{skew:+.2f}", border=True, height="stretch", help="0 for a symmetric distribution.")
    m[3].metric("Excess kurtosis", f"{kurt:.2f}", border=True, height="stretch", help="0 for a normal distribution; >0 = fat tails.")
    m[4].metric("Jarque–Bera p", "< 0.001" if jb_p < 1e-3 else f"{jb_p:.3f}", border=True, height="stretch", help="Normality test; p < 0.05 rejects normality.")

    ratio = tail_obs / tail_norm if tail_norm else float("nan")
    st.info(
        f"**Fat tails.** Moves beyond 3σ happened on **{tail_obs:.2f}%** of days versus "
        f"{tail_norm:.2f}% under a normal distribution — about **{ratio:.0f}× more often**. "
        "This is why forecast intervals are calibrated empirically instead of assuming normality.",
        icon=":material/insights:")

    a, b = st.columns(2)
    with a, st.container(border=True):
        card_title("Distribution of daily log returns", "Histogram vs fitted normal curve")
        chart(charts.returns_histogram(ret), key="ex_hist")
    with b, st.container(border=True):
        card_title("Q–Q plot", "Points leaving the diagonal at the ends = fat tails")
        chart(charts.qq_plot(ret), key="ex_qq")

# ── Volatility ───────────────────────────────────────────────────────
with tab_vol:
    vol21 = ret.rolling(21).std() * np.sqrt(252) * 100
    vol63 = ret.rolling(63).std() * np.sqrt(252) * 100
    with st.container(border=True):
        card_title("Rolling annualised volatility", "Volatility comes in regimes — calm and turbulent periods cluster",
                   data=pd.DataFrame({"21-day": vol21, "63-day": vol63}).dropna().round(2))
        chart(charts.line_chart({"21-day": vol21, "63-day": vol63},
                                {"21-day": ACCENT, "63-day": BENCHMARK_COLOR}, suffix="%", decimals=1),
              key="ex_vol")
    if ds.market is not None and "vix_close" in ds.market:
        vix = ds.market["vix_close"].reindex(prices.index).ffill()
        corr = float(vol21.corr(vix))
        with st.container(border=True):
            card_title("VIX (market fear gauge)", f"Correlation with this stock's 21-day volatility: ρ = {corr:.2f}")
            chart(charts.line_chart({"VIX": vix}, {"VIX": BENCHMARK_COLOR}, height=260, decimals=1), key="ex_vix")

# ── Autocorrelation ──────────────────────────────────────────────────
with tab_acf:
    nlags = 30
    conf = 1.96 / np.sqrt(len(ret))
    acf_ret = acf(ret, nlags=nlags, fft=True)
    acf_abs = acf(ret.abs(), nlags=nlags, fft=True)
    lb = acorr_ljungbox(ret, lags=[10], return_df=True)
    lb_abs = acorr_ljungbox(ret.abs(), lags=[10], return_df=True)
    sig_ret = int((np.abs(acf_ret[1:]) > conf).sum())
    sig_abs = int((np.abs(acf_abs[1:]) > conf).sum())

    st.info(
        f"**Direction is nearly unpredictable, magnitude is not.** Only {sig_ret} of {nlags} return "
        f"autocorrelations fall outside the 95% band (Ljung–Box p = {lb['lb_pvalue'].iloc[0]:.3f}), "
        f"while {sig_abs} of {nlags} absolute-return autocorrelations do (p = {lb_abs['lb_pvalue'].iloc[0]:.1e}). "
        "Expect models to struggle with next-day direction but capture volatility.",
        icon=":material/insights:")
    a, b = st.columns(2)
    with a, st.container(border=True):
        card_title("ACF of daily returns", "Shaded band = 95% confidence for zero correlation",
                   data=pd.DataFrame({"lag": range(1, nlags + 1), "acf": acf_ret[1:].round(4)}).set_index("lag"))
        chart(charts.acf_chart(acf_ret, conf, "Returns"), key="ex_acf1")
    with b, st.container(border=True):
        card_title("ACF of absolute returns", "Persistent correlation = volatility clustering",
                   data=pd.DataFrame({"lag": range(1, nlags + 1), "acf": acf_abs[1:].round(4)}).set_index("lag"))
        chart(charts.acf_chart(acf_abs, conf, "|Returns|", BENCHMARK_COLOR), key="ex_acf2")

# ── Stationarity ─────────────────────────────────────────────────────
with tab_stat:
    rows = []
    for name, series in (("Close price", prices["Close"]), ("Log price", np.log(prices["Close"])),
                         ("Daily log return", ret)):
        stat_, p, lags, n, crit, _ = adfuller(series.dropna(), autolag="AIC")
        rows.append({"Series": name, "ADF statistic": stat_, "p-value": p, "5% critical": crit["5%"],
                     "Verdict": "Stationary" if p < 0.05 else "Non-stationary (unit root)"})
    table = pd.DataFrame(rows).set_index("Series")
    st.info("**Why we model returns, not prices.** The Augmented Dickey–Fuller test cannot reject a unit "
            "root for the price level, so its mean and variance drift over time and a model trained on "
            "old price levels extrapolates badly. Log returns are stationary, so every model here "
            "predicts the next-day log return and converts it back to a price.", icon=":material/insights:")
    with st.container(border=True):
        card_title("Augmented Dickey–Fuller test", "H₀: the series has a unit root (non-stationary)")
        st.dataframe(table, column_config={
            "ADF statistic": st.column_config.NumberColumn(format="%.3f"),
            "p-value": st.column_config.NumberColumn(format="%.4f"),
            "5% critical": st.column_config.NumberColumn(format="%.3f"),
        })

# ── Feature signal ───────────────────────────────────────────────────
with tab_feat:
    data = labeled(frame)
    feats = feature_columns(data)
    ic = data[feats].corrwith(data["target"], method="spearman").sort_values(key=np.abs, ascending=False)
    labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in ic.index]
    st.info(f"**Signal is weak.** The strongest single feature has a rank correlation (information "
            f"coefficient) of only **{ic.abs().max():.3f}** with tomorrow's return. Values around 0.02–0.05 "
            "are typical even for professional factors — another reason to benchmark against the random walk.",
            icon=":material/insights:")
    a, b = st.columns([1, 1.15])
    with a, st.container(border=True):
        card_title("Information coefficient", "Spearman ρ between each feature and next-day return",
                   data=pd.DataFrame({"feature": labels, "IC": ic.values.round(4)}).set_index("feature"))
        chart(charts.signed_bars(ic, labels), key="ex_ic")
    with b, st.container(border=True):
        card_title("Feature correlation matrix", "Redundant features cluster; tree models cope, linear models need regularisation")
        chart(charts.correlation_heatmap(data[feats].corr()), key="ex_corr")

# ── Raw data ─────────────────────────────────────────────────────────
with tab_data:
    which = st.segmented_control("Dataset", ["Prices (OHLCV)", "Model features"], default="Prices (OHLCV)",
                                 required=True, label_visibility="collapsed")
    df = prices if which == "Prices (OHLCV)" else frame
    st.dataframe(df.sort_index(ascending=False), height=460)
    st.download_button("Download CSV", df.to_csv().encode(), f"{ticker}_{'prices' if df is prices else 'features'}.csv",
                       "text/csv", icon=":material/download:")

footer()
