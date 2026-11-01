"""Generate notebooks/01_eda.ipynb (run, then execute with nbconvert — see README)."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 01 · Exploratory data analysis\n\n"
       "**Question:** is there anything in a stock's own price history (plus market context) that "
       "predicts its next-day return — and what does the data imply for how we should model it?\n\n"
       "Each section ends with the modelling decision it motivates. The same analysis is available "
       "interactively on the app's *Explore* page for any ticker."),
    code("import sys, warnings\n"
         "from pathlib import Path\n"
         "sys.path.insert(0, str(Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from scipy import stats\n"
         "from statsmodels.tsa.stattools import adfuller, acf\n"
         "from statsmodels.stats.diagnostic import acorr_ljungbox\n\n"
         "from stocksense import pipeline\n"
         "from stocksense.config import load_config\n"
         "from stocksense.features import feature_columns, labeled, FEATURE_DESCRIPTIONS\n\n"
         "BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#898781'\n"
         "plt.rcParams.update({'figure.figsize': (11, 4), 'axes.spines.top': False, 'axes.spines.right': False,\n"
         "                     'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.6,\n"
         "                     'axes.edgecolor': '#c3c2b7', 'font.size': 10})\n\n"
         "TICKER, YEARS = 'AAPL', 5\n"
         "ds = pipeline.load_dataset(load_config(), TICKER, YEARS)\n"
         "prices, frame = ds.prices, ds.frame\n"
         "ret = np.log(prices['Close']).diff().dropna()\n"
         "print(f'{TICKER}: {len(prices)} trading days, {prices.index[0]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d}')\n"
         "prices.tail()"),

    md("## 1 · Price vs returns"),
    code("fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
         "a.plot(prices['Close'], color=BLUE, lw=1.6); a.set_title('Adjusted close'); a.set_ylabel('$')\n"
         "b.plot(ret * 100, color=GREY, lw=0.7); b.set_title('Daily log return'); b.set_ylabel('%')\n"
         "plt.tight_layout()"),
    md("The price trends; returns hover around zero with bursts of large moves. Prices and returns "
       "behave very differently, which the next two sections quantify."),

    md("## 2 · Stationarity (Augmented Dickey–Fuller)"),
    code("rows = []\n"
         "for name, s in [('Close', prices['Close']), ('Log price', np.log(prices['Close'])), ('Log return', ret)]:\n"
         "    stat, p, *_ = adfuller(s.dropna(), autolag='AIC')\n"
         "    rows.append({'series': name, 'ADF stat': round(stat, 3), 'p-value': round(p, 4),\n"
         "                 'stationary (5%)': p < 0.05})\n"
         "pd.DataFrame(rows).set_index('series')"),
    md("**Decision:** the price level has a unit root; log returns are stationary. All models predict "
       "the **next-day log return** and convert back to a price, instead of regressing on raw prices "
       "(the original project's approach, which mostly learns to copy yesterday's price)."),

    md("## 3 · Distribution of returns"),
    code("skew, kurt = stats.skew(ret), stats.kurtosis(ret)\n"
         "z = (ret - ret.mean()) / ret.std()\n"
         "print(f'ann. mean {ret.mean()*252:+.1%} | ann. vol {ret.std()*np.sqrt(252):.1%} | '\n"
         "      f'skew {skew:+.2f} | excess kurtosis {kurt:.2f} | JB p = {stats.jarque_bera(ret).pvalue:.1e}')\n"
         "print(f'|z| > 3 on {(z.abs() > 3).mean():.2%} of days vs {2*stats.norm.sf(3):.2%} under normality')\n\n"
         "fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))\n"
         "a.hist(ret * 100, bins=80, density=True, color=BLUE, alpha=0.75, edgecolor='white', linewidth=0.4)\n"
         "xs = np.linspace(ret.min() * 100, ret.max() * 100, 300)\n"
         "a.plot(xs, stats.norm.pdf(xs, ret.mean() * 100, ret.std() * 100), color=ORANGE, lw=2, label='Normal fit')\n"
         "a.set_title('Daily log returns (%)'); a.legend()\n"
         "stats.probplot(z, dist='norm', plot=b); b.get_lines()[0].set(color=BLUE, markersize=3, alpha=0.6)\n"
         "b.get_lines()[1].set(color=GREY); b.set_title('Q–Q plot vs normal')\n"
         "plt.tight_layout()"),
    md("**Decision:** returns are fat-tailed (excess kurtosis ≫ 0), so forecast intervals use "
       "**empirical (split-conformal) residual quantiles** instead of a Gaussian ±1.96σ."),

    md("## 4 · Autocorrelation: direction vs magnitude"),
    code("conf = 1.96 / np.sqrt(len(ret))\n"
         "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)\n"
         "for ax, series, title, color in [(axes[0], ret, 'ACF of returns', BLUE),\n"
         "                                 (axes[1], ret.abs(), 'ACF of |returns|', ORANGE)]:\n"
         "    v = acf(series, nlags=30, fft=True)[1:]\n"
         "    ax.bar(range(1, 31), v, color=color, width=0.6)\n"
         "    ax.axhspan(-conf, conf, color=GREY, alpha=0.18, lw=0); ax.set_title(title); ax.set_xlabel('lag (days)')\n"
         "plt.tight_layout()\n"
         "pd.DataFrame({'Ljung–Box p (10 lags)': [acorr_ljungbox(ret, lags=[10])['lb_pvalue'].iloc[0],\n"
         "                                         acorr_ljungbox(ret.abs(), lags=[10])['lb_pvalue'].iloc[0]]},\n"
         "             index=['returns', '|returns|'])"),
    md("Returns are close to white noise, while absolute returns are strongly persistent "
       "(**volatility clustering**). **Decision:** include realised-volatility and VIX features, and "
       "expect next-day *direction* to be hard to beat — hence the random-walk benchmark and the "
       "Diebold–Mariano test in the Model Lab."),

    md("## 5 · Volatility regimes and market context"),
    code("vol = ret.rolling(21).std() * np.sqrt(252) * 100\n"
         "fig, ax = plt.subplots()\n"
         "ax.plot(vol, color=BLUE, lw=1.4, label=f'{TICKER} 21-day vol')\n"
         "if ds.market is not None and 'vix_close' in ds.market:\n"
         "    vix = ds.market['vix_close'].reindex(vol.index).ffill()\n"
         "    ax.plot(vix, color=ORANGE, lw=1.2, label='VIX')\n"
         "    print(f'corr(21d vol, VIX) = {vol.corr(vix):.2f}')\n"
         "ax.set_ylabel('% annualised'); ax.legend(); ax.set_title('Volatility regimes')\n"
         "if ds.market is not None and 'mkt_close' in ds.market:\n"
         "    m = np.log(ds.market['mkt_close']).diff().reindex(ret.index)\n"
         "    beta = ret.cov(m) / m.var()\n"
         "    print(f'beta vs SPY = {beta:.2f}, correlation = {ret.corr(m):.2f}')"),
    md("Both series share units (% annualised volatility), so one axis is honest here. The stock's "
       "volatility tracks the VIX and its returns co-move with the market (beta ≈ 1). **Decision:** "
       "add SPY return/momentum/volatility and VIX level/change as context features."),

    md("## 6 · Feature signal (information coefficient)"),
    code("data = labeled(frame)\n"
         "feats = feature_columns(data)\n"
         "ic = data[feats].corrwith(data['target'], method='spearman').sort_values(key=np.abs)\n"
         "fig, ax = plt.subplots(figsize=(8, 7))\n"
         "ax.barh([FEATURE_DESCRIPTIONS.get(f, f) for f in ic.index], ic.values,\n"
         "        color=[BLUE if v >= 0 else '#e34948' for v in ic.values])\n"
         "ax.axvline(0, color=GREY, lw=0.8); ax.set_title('Spearman IC with next-day return')\n"
         "plt.tight_layout()\n"
         "print(f'max |IC| = {ic.abs().max():.3f} ({ic.abs().idxmax()})')"),
    md("Every individual feature is weakly related to tomorrow's return (|IC| of a few hundredths). "
       "**Decision:** combine features with regularised models (Ridge, shallow XGBoost with early "
       "stopping) and judge them only on walk-forward, out-of-sample performance."),

    md("## 7 · Summary → modelling choices\n\n"
       "| Finding | Modelling consequence |\n|---|---|\n"
       "| Prices have a unit root; returns are stationary | Predict next-day **log return**, convert to price |\n"
       "| Fat tails | **Conformal** intervals from empirical residuals |\n"
       "| Returns ≈ white noise | **Random-walk benchmark** + Diebold–Mariano significance test |\n"
       "| Volatility clusters, tracks VIX | Realised-vol, VIX and market features |\n"
       "| Weak individual signals | Regularised models, walk-forward evaluation, no in-sample claims |\n\n"
       "Next: run the models — `python -m scripts.train --ticker AAPL --track` or the app's *Model Lab*."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("01_eda.ipynb")
nbf.write(nb, out)
print(f"wrote {out}")
