"""Train, evaluate, backtest and forecast from the command line.

Examples
--------
    python -m scripts.train --ticker AAPL
    python -m scripts.train --ticker MSFT --years 8 --models ridge xgboost --track
"""

from __future__ import annotations

import argparse
import sys
import warnings

from stocksense import pipeline
from stocksense.config import load_config
from stocksense.evaluation import verdict
from stocksense.features import feature_columns, labeled
from stocksense.models import model_factory
from stocksense.tracking import log_to_mlflow, save_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", help="Yahoo Finance symbol (default: from config)")
    ap.add_argument("--years", type=int, help="Years of history (default: from config)")
    ap.add_argument("--models", nargs="+", choices=pipeline.DEFAULT_MODELS,
                    help="Models to evaluate (the random walk is always included)")
    ap.add_argument("--config", help="Path to a YAML config (default: configs/default.yaml)")
    ap.add_argument("--track", action="store_true", help="Log the run to MLflow (mlflow.db)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    cfg = load_config(args.config)
    ds = pipeline.load_dataset(cfg, args.ticker, args.years)
    print(f"▸ {ds.ticker}: {len(ds.prices)} trading days "
          f"({ds.prices.index[0]:%Y-%m-%d} → {ds.prices.index[-1]:%Y-%m-%d}), "
          f"{len(feature_columns(ds.frame))} features")

    def progress(frac: float, msg: str) -> None:
        sys.stdout.write(f"\r  walk-forward {frac:5.0%}  {msg:<40}")
        sys.stdout.flush()

    result = pipeline.evaluate(ds, cfg, args.models, progress)
    print()
    board = result.leaderboard
    cols = ["name", "rmse_price", "mape", "rmse_ret", "dir_acc", "r2_oos", "dm_p"]
    print("\nOut-of-sample leaderboard (walk-forward, %d folds):" % len(result.folds))
    print(board[cols].round(4).to_string())
    print("\n" + verdict(board)[1].replace("**", ""))

    best = board.drop(index="naive").index[0]
    bt = pipeline.backtest(result, best, cfg)
    print(f"\nBacktest of {board.loc[best, 'name']} (long/flat, {cfg.backtest.cost_bps:g} bps costs):")
    for label, s in (("strategy", bt.strategy), ("buy & hold", bt.benchmark)):
        print(f"  {label:<11} return {s['total_return']:7.1f}%  sharpe {s['sharpe']:5.2f}  "
              f"max DD {s['max_drawdown']:6.1f}%")

    fc = pipeline.forecast(ds, cfg)
    t = fc.table
    print(f"\n{len(t)}-day forecast from {fc.last_date:%Y-%m-%d} close ${fc.last_close:,.2f}:")
    print(t[["date", "pred", "lo_80", "hi_80"]].round({"pred": 2, "lo_80": 2, "hi_80": 2})
          .to_string(index=False))

    # Refit the shortlisted models on all labeled history and persist them.
    data, feats = labeled(ds.frame), feature_columns(ds.frame)
    final_models = {k: model_factory(k, cfg)().fit(data, feats)
                    for k in board.index if k in ("ridge", "xgboost", "lstm")}
    run_dir = save_run(ds.ticker, cfg, {
        "leaderboard": board, "predictions": result.predictions, "fold_scores": result.fold_scores,
        "forecast": t, "backtest_daily": bt.daily,
        "backtest_summary": {"model": best, "strategy": bt.strategy, "benchmark": bt.benchmark},
    }, final_models)
    print(f"\n✔ Artifacts saved to {run_dir}")

    if args.track:
        extra = {f"backtest.{k}": v for k, v in bt.strategy.items()}
        run_id = log_to_mlflow(ds.ticker, cfg, board, extra, run_dir)
        print(f"✔ Logged to MLflow (run {run_id}). View with: mlflow ui --backend-store-uri sqlite:///mlflow.db"
              if run_id else "! mlflow not installed — skipped tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
