import copy

import numpy as np
import pandas as pd
import streamlit as st

from stocksense import pipeline
from stocksense.evaluation import verdict, walk_forward_splits
from stocksense.features import FEATURE_DESCRIPTIONS, labeled
from stocksense.models import MODEL_CLASSES, MODEL_NAMES
from ui import charts, state
from ui.components import card_title, chart, footer
from ui.theme import CURRENCY_SYMBOLS

ticker, years = state.selection()
ds = state.current_dataset()
cfg = copy.deepcopy(state.config())  # never mutate the cached config
info = state.company(ticker)
sym = CURRENCY_SYMBOLS.get(info.get("currency", "USD"), "")
data = labeled(ds.frame)

st.markdown("### Model Lab")
st.caption("Every model is trained only on the past and scored on the future it never saw, "
           "using an expanding-window walk-forward backtest.")

if years < state.MIN_YEARS_FOR_MODELS:
    st.warning("Choose at least 2 years of history for a meaningful walk-forward evaluation.",
               icon=":material/warning:")
    st.stop()

# ── Configuration ────────────────────────────────────────────────────
with st.container(border=True):
    c1, c2, c3 = st.columns([2.6, 1, 0.8], vertical_alignment="bottom")
    with c1:
        choices = [k for k in MODEL_CLASSES if k != "naive"]
        models = st.pills("Models (the random walk benchmark is always included)", choices,
                          selection_mode="multi", default=choices, format_func=MODEL_NAMES.get,
                          key="lab_models")
    with c2:
        n_folds = st.slider("Walk-forward folds", 3, 8, cfg.evaluation.n_folds, key="lab_folds")
    with c3:
        run = st.button("Run evaluation", type="primary", icon=":material/play_arrow:", width="stretch")
    if "lstm" in (models or []):
        st.caption(":material/schedule: LSTM retrains once per fold — expect ~20–40 s in total.")

try:
    folds = walk_forward_splits(len(data), n_folds, cfg.evaluation.initial_train_frac)
except ValueError as exc:
    st.warning(str(exc), icon=":material/warning:")
    st.stop()

if run:
    cfg.evaluation.n_folds = n_folds
    bar = st.progress(0.0, text="Starting…")
    result = pipeline.evaluate(ds, cfg, ["naive", *(models or [])],
                               progress=lambda f, msg: bar.progress(f, text=msg))
    bar.empty()
    state.store_lab_result(result)

result = state.lab_result()

if result is None:
    with st.container(border=True):
        card_title("How the evaluation works",
                   f"{len(data):,} labelled trading days · {n_folds} folds · each fold retrains on all prior data")
        chart(charts.fold_diagram(data.index, folds), key="lab_folds_diagram")
    st.info("Press **Run evaluation** to train and score the models.", icon=":material/play_circle:")
    footer()
    st.stop()

board = result.leaderboard
tone, message = verdict(board)
(st.success if tone == "good" else st.info)(message, icon=":material/gavel:")

# ── Headline metrics ────────────────────────────────────────────────
best = board.drop(index="naive").index[0]
b, n = board.loc[best], board.loc["naive"]
k = st.columns(4)
k[0].metric("Best model (return RMSE)", b["name"], border=True, height="stretch")
k[1].metric("Error vs random walk", f"{(b['rmse_ret'] / n['rmse_ret'] - 1) * 100:+.2f}%",
            delta="lower is better", delta_color="off", delta_arrow="off", border=True, height="stretch",
            help="Relative change in out-of-sample return RMSE versus predicting zero return.")
k[2].metric("Directional accuracy", f"{b['dir_acc']:.1f}%",
            delta=f"{b['dir_acc'] - n['dir_acc']:+.1f} pts vs always-up", border=True, height="stretch",
            help="Share of days where the predicted sign matched the realised sign.")
k[3].metric("Diebold–Mariano p-value", f"{b['dm_p']:.3f}", border=True, height="stretch",
            delta="significant" if b["beats_benchmark"] else "not significant",
            delta_color="normal" if b["beats_benchmark"] else "off", delta_arrow="off",
            help="Tests whether the model's squared errors are lower than the random walk's. p < 0.05 = significant.")

# ── Leaderboard ─────────────────────────────────────────────────────
with st.container(border=True):
    card_title("Out-of-sample leaderboard",
               f"{len(result.predictions):,} test days across {len(result.folds)} folds · sorted by return RMSE")
    view = board.assign(rmse_ret_bps=board["rmse_ret"] * 1e4,
                        dm_p=[("benchmark" if k == "naive" else f"{p:.3f}") for k, p in board["dm_p"].items()])
    st.dataframe(
        view[["name", "rmse_price", "mape", "rmse_ret_bps", "dir_acc", "r2_oos", "dm_p", "beats_benchmark"]],
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Model"),
            "rmse_price": st.column_config.NumberColumn("Price RMSE", format=f"{sym}%.2f",
                                                        help="Error of the implied next-day price"),
            "mape": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
            "rmse_ret_bps": st.column_config.NumberColumn("Return RMSE (bps)", format="%.1f",
                                                          help="1 bp = 0.01 %. The metric models are ranked on."),
            "dir_acc": st.column_config.ProgressColumn("Directional acc.", format="%.1f%%", min_value=40, max_value=60),
            "r2_oos": st.column_config.NumberColumn("R² OOS", format="%.2f%%",
                                                    help="Campbell–Thompson out-of-sample R² vs zero-return forecast"),
            "dm_p": st.column_config.TextColumn("DM p-value",
                                                help="Diebold–Mariano test vs the random walk; < 0.05 = significant"),
            "beats_benchmark": st.column_config.CheckboxColumn("Beats random walk"),
        },
    )

# ── Diagnostics ─────────────────────────────────────────────────────
tabs = ["Predictions", "Errors by fold", "Feature importance"] + (["LSTM training"] if "lstm" in result.histories else [])
t_pred, t_fold, t_imp, *t_lstm = st.tabs(tabs)

with t_pred:
    preds = result.predictions
    pos = ds.frame.index.get_indexer(preds.index)
    next_dates = ds.frame.index[np.minimum(pos + 1, len(ds.frame) - 1)]
    c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="bottom")
    with c1:
        shown = st.pills("Models shown", [m for m in result.models if m != "naive"],
                         selection_mode="multi", default=[best], format_func=MODEL_NAMES.get, key="lab_shown")
    with c2:
        mode = st.segmented_control("View", ["Price", "Return"], default="Price", required=True, key="lab_mode")
    with c3:
        span = st.segmented_control("Window", ["3M", "6M", "All"], default="3M", required=True, key="lab_span")
    n_days = {"3M": 63, "6M": 126, "All": len(preds)}[span]
    sl = slice(-n_days, None)
    if mode == "Price":
        actual = (preds["close"] * np.exp(preds["actual"])).iloc[sl]
        series = {m: (preds["close"] * np.exp(preds[m])).iloc[sl] for m in ["naive", *(shown or [])]}
    else:
        actual = (preds["actual"] * 100).iloc[sl]
        series = {m: (preds[m] * 100).iloc[sl] for m in (shown or [])}
    with st.container(border=True):
        cap = ("Implied next-day price; the random walk (grey) is simply yesterday's close"
               if mode == "Price" else "Predicted vs realised daily log return — note how small the predictions are")
        table = pd.DataFrame({"actual": actual, **{MODEL_NAMES[k]: v for k, v in series.items()}})
        table.index = next_dates[sl]
        card_title("Actual vs predicted", cap, data=table.round(4))
        chart(charts.predictions_chart(next_dates[sl], actual.values, {k: v.values for k, v in series.items()},
                                       mode, sym), key="lab_pred")

with t_fold:
    with st.container(border=True):
        pivot = result.fold_scores.pivot(index="fold", columns="model", values="rmse_ret")
        card_title("Improvement over the random walk, per fold",
                   "Bars above zero beat the benchmark in that period; consistency matters more than one good fold",
                   data=((1 - pivot.div(pivot["naive"], axis=0)) * 100).round(3).rename(columns=MODEL_NAMES))
        chart(charts.fold_improvement_chart(result.fold_scores), key="lab_fold")

with t_imp:
    with st.spinner("Computing SHAP values…"):
        imp = state.xgb_importance(ticker, years).head(15)
    labels = [FEATURE_DESCRIPTIONS.get(f, f) for f in imp.index]
    with st.container(border=True):
        card_title("What XGBoost relies on", "Mean |SHAP value| per feature (model refit on all history), in bps of return",
                   data=pd.DataFrame({"feature": labels, "mean |SHAP| (bps)": (imp.values * 1e4).round(3)}).set_index("feature"))
        chart(charts.signed_bars(imp * 1e4, labels, suffix=" bps", decimals=2, signed_colors=False), key="lab_imp")

if t_lstm:
    with t_lstm[0], st.container(border=True):
        card_title("LSTM learning curves", "One line per fold (later folds drawn darker); "
                   "early stopping restores the best validation epoch")
        chart(charts.loss_curves(result.histories["lstm"]), key="lab_loss")

st.page_link("app_pages/backtest.py", label="Trade these forecasts in the Backtest", icon=":material/arrow_forward:")
footer()
