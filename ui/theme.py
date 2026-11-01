"""Design tokens shared by every chart and page (dark mode).

Categorical colours are the validated dark-surface palette (see README);
each model keeps its colour on every page so identity never shifts.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

SURFACE = "#141413"
CARD = "#1a1a19"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

UP = "#0ca30c"
DOWN = "#e66767"
ACCENT = "#3987e5"
NEG_POLE = "#e66767"   # diverging: negative
POS_POLE = "#3987e5"   # diverging: positive
MID = "#383835"        # diverging neutral midpoint

MODEL_COLORS = {
    "xgboost": "#3987e5",
    "ridge": "#d95926",
    "arima": "#199e70",
    "lstm": "#c98500",
    "drift": "#d55181",
    "naive": MUTED,     # the benchmark is deliberately recessive
}
BENCHMARK_COLOR = "#c98500"   # buy & hold in the backtest

FONT = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'

CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
                    "CAD": "C$", "AUD": "A$", "HKD": "HK$", "CNY": "¥", "CHF": "CHF "}


def inject_css() -> None:
    css = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    st.html(f"<style>{css.read_text()}</style>")
