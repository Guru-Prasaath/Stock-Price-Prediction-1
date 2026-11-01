"""Plotly figure builders following one visual system.

Rules: hairline recessive grid, 2px lines, one y-axis per chart, fixed colour
per entity, legends for >= 2 series plus selective end-of-line labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from stocksense.models import MODEL_NAMES

from .theme import (ACCENT, AXIS, BENCHMARK_COLOR, DOWN, FONT, GRID, INK, INK_2, MID,
                    MODEL_COLORS, MUTED, NEG_POLE, POS_POLE, UP)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _axis(**kw) -> dict:
    return dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS, showline=False, automargin=True,
                tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=MUTED, size=11)),
                **kw)


def layout(height: int = 380, hovermode: str = "x unified", legend: bool = True, **kw) -> dict:
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=12),
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor="#232321", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                    font=dict(color=INK_2, size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=_axis(showgrid=False),
        yaxis=_axis(side="right"),
        barcornerradius=4,
        bargap=0.25,
    )
    base.update(kw)
    return base


def _end_label(fig: go.Figure, x, y, text: str, color: str, row=None, col=None) -> None:
    """Direct label at the last point of a series (identity not by colour alone)."""
    kw = dict(row=row, col=col) if row else {}
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>", showarrow=False, xanchor="left",
                       xshift=6, font=dict(color=INK, size=11), bgcolor=_rgba(color, 0.18),
                       bordercolor=color, borderwidth=1, borderpad=3, **kw)


# ───────────────────────────── Overview ─────────────────────────────

def price_chart(prices: pd.DataFrame, kind: str = "Candles", overlays=(), currency="$") -> go.Figure:
    p = prices
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                        vertical_spacing=0.04)
    if kind == "Candles":
        fig.add_trace(go.Candlestick(
            x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], name="Price",
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=p.index, y=p["Close"], name="Close", mode="lines",
                                 line=dict(color=ACCENT, width=2), fill="tozeroy",
                                 fillcolor=_rgba(ACCENT, 0.08),
                                 hovertemplate=f"{currency}%{{y:,.2f}}<extra>Close</extra>"),
                      row=1, col=1)

    full_close = prices["Close"]
    overlay_styles = {"SMA 20": ("#c98500", 20), "SMA 50": ("#d55181", 50), "SMA 200": ("#199e70", 200)}
    for name in overlays:
        if name in overlay_styles:
            color, w = overlay_styles[name]
            sma = full_close.rolling(w).mean().reindex(p.index)
            fig.add_trace(go.Scatter(x=p.index, y=sma, name=name, mode="lines",
                                     line=dict(color=color, width=1.5),
                                     hovertemplate=f"{currency}%{{y:,.2f}}<extra>{name}</extra>"),
                          row=1, col=1)
        elif name == "Bollinger":
            mid = full_close.rolling(20).mean()
            sd = full_close.rolling(20).std()
            for band, label in ((mid + 2 * sd, "Upper band"), (mid - 2 * sd, "Lower band")):
                fig.add_trace(go.Scatter(x=p.index, y=band.reindex(p.index), name=label, mode="lines",
                                         line=dict(color=MUTED, width=1), showlegend=False,
                                         fill="tonexty" if label == "Lower band" else None,
                                         fillcolor=_rgba(MUTED, 0.08),
                                         hovertemplate=f"{currency}%{{y:,.2f}}<extra>{label}</extra>"),
                              row=1, col=1)

    up = p["Close"] >= p["Open"]
    fig.add_trace(go.Bar(x=p.index, y=p["Volume"], name="Volume", showlegend=False,
                         marker=dict(color=np.where(up, _rgba(UP, 0.55), _rgba(DOWN, 0.55)), line_width=0),
                         hovertemplate="%{y:,.0f}<extra>Volume</extra>"), row=2, col=1)

    fig.update_layout(**layout(height=520, legend=bool(overlays)))
    fig.update_layout(bargap=0.1, barcornerradius=1)
    fig.update_xaxes(**_axis(showgrid=False), rangeslider_visible=False,
                     rangebreaks=[dict(bounds=["sat", "mon"])] if len(p) < 400 else [])
    fig.update_yaxes(**_axis(side="right"))
    fig.update_yaxes(tickprefix=currency, row=1, col=1)
    fig.update_yaxes(showgrid=False, tickformat="~s", row=2, col=1)
    return fig


def period_returns_chart(returns: pd.Series) -> go.Figure:
    """Signed horizontal bars; values live in the category labels so they never collide."""
    colors = [POS_POLE if v >= 0 else NEG_POLE for v in returns.values]
    labels = [f"{k}  <b>{v:+.1f}%</b>" for k, v in returns.items()]
    fig = go.Figure(go.Bar(
        x=returns.values, y=labels, orientation="h", marker=dict(color=colors, line_width=0),
        hovertemplate="%{x:+.2f}%<extra></extra>"))
    fig.update_layout(**layout(height=250, hovermode="closest", legend=False))
    fig.update_yaxes(autorange="reversed", showgrid=False, side="left", tickfont=dict(color=INK_2, size=12))
    fig.update_xaxes(showgrid=True, ticksuffix="%", zeroline=True, zerolinecolor=AXIS, nticks=4)
    return fig


# ───────────────────────────── Explore ──────────────────────────────

def returns_histogram(ret: pd.Series) -> go.Figure:
    r = ret.dropna() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=80, histnorm="probability density", name="Daily returns",
                               marker=dict(color=_rgba(ACCENT, 0.75), line=dict(color="#141413", width=1)),
                               hovertemplate="%{x:.2f}%<br>density %{y:.3f}<extra></extra>"))
    xs = np.linspace(r.min(), r.max(), 300)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, r.mean(), r.std()), name="Normal fit",
                             mode="lines", line=dict(color=BENCHMARK_COLOR, width=2),
                             hovertemplate="%{x:.2f}%<extra>Normal</extra>"))
    fig.update_layout(**layout(height=340, hovermode="closest", bargap=0.02, barcornerradius=0))
    fig.update_xaxes(ticksuffix="%", title_text="Daily log return")
    fig.update_yaxes(title_text="Density")
    return fig


def qq_plot(ret: pd.Series) -> go.Figure:
    r = np.sort(((ret.dropna() - ret.mean()) / ret.std()).to_numpy())
    theo = stats.norm.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
    lim = float(max(abs(theo).max(), abs(r).max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="Normal",
                             line=dict(color=MUTED, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=theo, y=r, mode="markers", name="Observed",
                             marker=dict(color=ACCENT, size=5, opacity=0.6),
                             hovertemplate="theoretical %{x:.2f}σ<br>observed %{y:.2f}σ<extra></extra>"))
    fig.update_layout(**layout(height=340, hovermode="closest"))
    fig.update_xaxes(title_text="Normal quantile (σ)", showgrid=True)
    fig.update_yaxes(title_text="Observed quantile (σ)")
    return fig


def line_chart(series: dict[str, pd.Series], colors: dict[str, str], height=320, suffix="",
               prefix="", decimals=2, end_labels=True) -> go.Figure:
    fig = go.Figure()
    for name, s in series.items():
        s = s.dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(color=colors[name], width=2),
                                 hovertemplate=f"{prefix}%{{y:,.{decimals}f}}{suffix}<extra>{name}</extra>"))
        if end_labels and len(series) > 1 and len(s):
            _end_label(fig, s.index[-1], s.iloc[-1], name, colors[name])
    fig.update_layout(**layout(height=height, legend=len(series) > 1))
    fig.update_yaxes(ticksuffix=suffix, tickprefix=prefix)
    if end_labels and len(series) > 1:
        fig.update_layout(margin=dict(l=8, r=110, t=36, b=8))
    return fig


def acf_chart(values: np.ndarray, conf: float, title: str, color: str = ACCENT) -> go.Figure:
    lags = np.arange(1, len(values))
    v = values[1:]
    fig = go.Figure()
    fig.add_hrect(y0=-conf, y1=conf, fillcolor=_rgba(MUTED, 0.12), line_width=0)
    fig.add_trace(go.Bar(x=lags, y=v, name=title,
                         marker=dict(color=[color if abs(x) > conf else _rgba(color, 0.35) for x in v],
                                     line_width=0),
                         hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(**layout(height=280, hovermode="closest", legend=False, bargap=0.35))
    fig.update_xaxes(title_text="Lag (trading days)")
    fig.update_yaxes(range=[min(-0.15, v.min() * 1.2), max(0.15, v.max() * 1.2)], zeroline=True,
                     zerolinecolor=AXIS)
    return fig


def signed_bars(values: pd.Series, labels: list[str] | None = None, height: int | None = None,
                suffix: str = "", decimals: int = 3, signed_colors: bool = True) -> go.Figure:
    """Horizontal bars, largest |value| on top; blue = positive, red = negative."""
    v = values.iloc[::-1]
    labels = (labels or list(values.index))[::-1]
    colors = [(POS_POLE if x >= 0 else NEG_POLE) if signed_colors else ACCENT for x in v.values]
    fig = go.Figure(go.Bar(
        x=v.values, y=labels, orientation="h", marker=dict(color=colors, line_width=0),
        customdata=list(v.index),
        hovertemplate=f"%{{y}}<br>%{{x:+.{decimals}f}}{suffix}<extra>%{{customdata}}</extra>"))
    fig.update_layout(**layout(height=height or max(260, 26 * len(v) + 40), hovermode="closest",
                               legend=False, bargap=0.3))
    fig.update_yaxes(side="left", showgrid=False, tickfont=dict(color=INK_2, size=12))
    fig.update_xaxes(showgrid=True, zeroline=True, zerolinecolor=AXIS, ticksuffix=suffix)
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    scale = [[0, NEG_POLE], [0.5, MID], [1, POS_POLE]]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale=scale,
        xgap=2, ygap=2, colorbar=dict(thickness=10, tickfont=dict(color=MUTED, size=10), outlinewidth=0),
        hovertemplate="%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>"))
    fig.update_layout(**layout(height=560, hovermode="closest", legend=False))
    fig.update_xaxes(showgrid=False, tickangle=-45, tickfont=dict(size=10, color=MUTED))
    fig.update_yaxes(showgrid=False, side="left", autorange="reversed", tickfont=dict(size=10, color=MUTED))
    return fig


# ───────────────────────────── Model Lab ────────────────────────────

def fold_diagram(dates: pd.DatetimeIndex, folds) -> go.Figure:
    """Gantt-style view of the expanding-window walk-forward scheme."""
    fig = go.Figure()
    for f in folds:
        y = f"Fold {f.number}"
        t0, t1 = dates[0], dates[f.train_end - 1]
        s0, s1 = dates[f.test_start], dates[f.test_end - 1]
        fig.add_trace(go.Scatter(x=[t0, t1], y=[y, y], mode="lines", line=dict(color=_rgba(ACCENT, 0.45), width=14),
                                 name="Train", showlegend=f.number == 1,
                                 hovertemplate=f"Train: {t0:%b %Y} → {t1:%b %Y}<extra>{y}</extra>"))
        fig.add_trace(go.Scatter(x=[s0, s1], y=[y, y], mode="lines", line=dict(color=BENCHMARK_COLOR, width=14),
                                 name="Test (out-of-sample)", showlegend=f.number == 1,
                                 hovertemplate=f"Test: {s0:%b %Y} → {s1:%b %Y}<extra>{y}</extra>"))
    fig.update_layout(**layout(height=60 + 34 * len(folds), hovermode="closest"))
    fig.update_yaxes(autorange="reversed", showgrid=False, side="left", tickfont=dict(color=INK_2))
    fig.update_xaxes(showgrid=True)
    return fig


def predictions_chart(dates, actual: pd.Series, preds: dict[str, pd.Series], mode: str,
                      currency: str = "$") -> go.Figure:
    fig = go.Figure()
    if mode == "Price":
        fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual", mode="lines",
                                 line=dict(color=INK, width=2),
                                 hovertemplate=f"{currency}%{{y:,.2f}}<extra>Actual</extra>"))
        tmpl, suffix, prefix = f"{currency}%{{y:,.2f}}", "", currency
    else:
        fig.add_trace(go.Bar(x=dates, y=actual, name="Actual return",
                             marker=dict(color=_rgba(MUTED, 0.45), line_width=0),
                             hovertemplate="%{y:+.2f}%<extra>Actual</extra>"))
        tmpl, suffix, prefix = "%{y:+.3f}%", "%", ""
    for key, s in preds.items():
        name = MODEL_NAMES.get(key, key)
        fig.add_trace(go.Scatter(x=dates, y=s, name=name, mode="lines",
                                 line=dict(color=MODEL_COLORS.get(key, ACCENT), width=2),
                                 hovertemplate=f"{tmpl}<extra>{name}</extra>"))
    fig.update_layout(**layout(height=420, bargap=0.15, barcornerradius=1))
    fig.update_yaxes(tickprefix=prefix, ticksuffix=suffix)
    return fig


def fold_improvement_chart(fold_scores: pd.DataFrame, benchmark: str = "naive") -> go.Figure:
    """Return-RMSE improvement over the random walk per fold (positive = better)."""
    pivot = fold_scores.pivot(index="fold", columns="model", values="rmse_ret")
    imp = (1 - pivot.div(pivot[benchmark], axis=0)) * 100
    fig = go.Figure()
    for key in [c for c in imp.columns if c != benchmark]:
        fig.add_trace(go.Bar(x=[f"Fold {i}" for i in imp.index], y=imp[key], name=MODEL_NAMES.get(key, key),
                             marker=dict(color=MODEL_COLORS.get(key, ACCENT), line_width=0),
                             hovertemplate="%{y:+.2f}%<extra>" + MODEL_NAMES.get(key, key) + "</extra>"))
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(**layout(height=360, hovermode="x unified", bargap=0.25, bargroupgap=0.12))
    fig.update_yaxes(ticksuffix="%", title_text="RMSE improvement vs random walk")
    return fig


def loss_curves(histories: list[dict]) -> go.Figure:
    fig = go.Figure()
    for i, h in enumerate(histories, start=1):
        epochs = list(range(1, len(h["loss"]) + 1))
        fig.add_trace(go.Scatter(x=epochs, y=h["loss"], mode="lines", name="Train loss",
                                 legendgroup="train", showlegend=i == 1,
                                 line=dict(color=ACCENT, width=1.5), opacity=0.4 + 0.6 * i / len(histories),
                                 hovertemplate=f"fold {i} · epoch %{{x}}: %{{y:.3f}}<extra>train</extra>"))
        if "val_loss" in h:
            fig.add_trace(go.Scatter(x=epochs, y=h["val_loss"], mode="lines", name="Validation loss",
                                     legendgroup="val", showlegend=i == 1,
                                     line=dict(color=BENCHMARK_COLOR, width=1.5),
                                     opacity=0.4 + 0.6 * i / len(histories),
                                     hovertemplate=f"fold {i} · epoch %{{x}}: %{{y:.3f}}<extra>validation</extra>"))
    fig.update_layout(**layout(height=320, hovermode="closest"))
    fig.update_xaxes(title_text="Epoch", showgrid=True)
    fig.update_yaxes(title_text="MSE (scaled)")
    return fig


# ───────────────────────────── Forecast ─────────────────────────────

def fan_chart(history: pd.Series, table: pd.DataFrame, levels: list[float], currency="$") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", mode="lines",
                             line=dict(color=INK, width=2),
                             hovertemplate=f"{currency}%{{y:,.2f}}<extra>Close</extra>"))
    anchor_x = [history.index[-1]]
    anchor_y = [float(history.iloc[-1])]
    x = anchor_x + list(table["date"])
    for lvl, alpha in sorted(zip(levels, (0.32, 0.16)), key=lambda t: -t[0]):
        tag = int(lvl * 100)
        hi = anchor_y + list(table[f"hi_{tag}"])
        lo = anchor_y + list(table[f"lo_{tag}"])
        fig.add_trace(go.Scatter(x=x + x[::-1], y=hi + lo[::-1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor=_rgba(ACCENT, alpha),
                                 name=f"{tag}% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=anchor_y + list(table["pred"]), name="Forecast", mode="lines+markers",
                             line=dict(color=ACCENT, width=2), marker=dict(size=8, color=ACCENT,
                                                                           line=dict(color="#141413", width=2)),
                             hovertemplate=f"{currency}%{{y:,.2f}}<extra>Forecast</extra>"))
    fig.add_hline(y=anchor_y[0], line=dict(color=MUTED, width=1),
                  annotation=dict(text="Last close", font=dict(color=MUTED, size=11), xanchor="left"),
                  annotation_position="top left")
    fig.add_vline(x=history.index[-1], line=dict(color=AXIS, width=1))
    fig.update_layout(**layout(height=460))
    fig.update_yaxes(tickprefix=currency)
    return fig


# ───────────────────────────── Backtest ─────────────────────────────

def equity_chart(daily: pd.DataFrame, strategy_name: str) -> go.Figure:
    series = {strategy_name: (daily["strategy_equity"] - 1) * 100,
              "Buy & hold": (daily["benchmark_equity"] - 1) * 100}
    colors = {strategy_name: ACCENT, "Buy & hold": BENCHMARK_COLOR}
    fig = line_chart(series, colors, height=380, suffix="%", decimals=1)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    return fig


def drawdown_chart(daily: pd.DataFrame, strategy_name: str) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("benchmark_dd", "Buy & hold", BENCHMARK_COLOR),
                             ("strategy_dd", strategy_name, ACCENT)):
        fig.add_trace(go.Scatter(x=daily.index, y=daily[col] * 100, name=name, mode="lines",
                                 line=dict(color=color, width=1.5), fill="tozeroy",
                                 fillcolor=_rgba(color, 0.12),
                                 hovertemplate="%{y:.1f}%<extra>" + name + "</extra>"))
    fig.update_layout(**layout(height=260))
    fig.update_yaxes(ticksuffix="%")
    return fig


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    t = table.reindex(columns=range(1, 13))
    lim = float(np.nanmax(np.abs(t.values))) if np.isfinite(t.values).any() else 1.0
    text = [[("" if np.isnan(v) else f"{v:+.1f}") for v in row] for row in t.values]
    fig = go.Figure(go.Heatmap(
        z=t.values, x=months, y=[str(y) for y in t.index], zmin=-lim, zmax=lim,
        colorscale=[[0, NEG_POLE], [0.5, MID], [1, POS_POLE]], xgap=2, ygap=2,
        text=text, texttemplate="%{text}", textfont=dict(color=INK, size=11),
        colorbar=dict(thickness=10, ticksuffix="%", tickfont=dict(color=MUTED, size=10), outlinewidth=0),
        hovertemplate="%{x} %{y}: %{z:+.2f}%<extra></extra>"))
    fig.update_layout(**layout(height=70 + 34 * len(t), hovermode="closest", legend=False))
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(color=INK_2))
    fig.update_yaxes(showgrid=False, side="left", autorange="reversed", tickfont=dict(color=INK_2))
    return fig
