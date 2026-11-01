"""Reusable page building blocks and number formatting."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from .theme import CURRENCY_SYMBOLS

PLOTLY_CONFIG = {"displayModeBar": False, "scrollZoom": False}


def money(x: float | None, currency: str = "USD", decimals: int = 2) -> str:
    """Formatted price for markdown/metric text ("$" escaped so it never starts LaTeX)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ").replace("$", "\\$")
    return f"{sym}{x:,.{decimals}f}"


def big_money(x: float | None, currency: str = "USD") -> str:
    if not x:
        return "—"
    sym = CURRENCY_SYMBOLS.get(currency, f"{currency} ").replace("$", "\\$")
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{sym}{x / div:,.2f}{unit}"
    return f"{sym}{x:,.0f}"


def pct(x: float | None, decimals: int = 1, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%"


def card_title(title: str, caption: str | None = None, data: pd.DataFrame | None = None,
               data_label: str = "Data") -> None:
    """Card heading with an optional data-table popover (the chart's table twin)."""
    with st.container(horizontal=True, vertical_alignment="center", gap="medium"):
        st.markdown(f'<p class="ss-card-title">{title}</p>'
                    + (f'<p class="ss-card-caption">{caption}</p>' if caption else ""),
                    unsafe_allow_html=True, width="stretch")
        if data is not None:
            with st.popover(data_label, icon=":material/table_view:", width="content"):
                st.dataframe(data, height=320)


def chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def empty_state(icon: str, title: str, body: str, page: str | None = None,
                label: str | None = None) -> None:
    with st.container(border=True, horizontal_alignment="center"):
        st.space("medium")
        st.markdown(f"## {icon}", text_alignment="center")
        st.markdown(f"#### {title}", text_alignment="center")
        st.markdown(f":gray[{body}]", text_alignment="center", width=620)
        if page:
            st.page_link(page, label=label or "Open", icon=":material/arrow_forward:")
        st.space("medium")


def footer() -> None:
    st.markdown(
        '<div class="ss-footer">StockSense · educational project, not investment advice · '
        "data from Yahoo Finance via yfinance</div>", unsafe_allow_html=True)
