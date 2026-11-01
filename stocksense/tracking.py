"""Run artifacts on disk + optional MLflow experiment tracking."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import __version__
from .config import ROOT, Config

ARTIFACTS_DIR = ROOT / "artifacts"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def save_run(ticker: str, cfg: Config, outputs: dict, models: dict | None = None) -> Path:
    """Write a self-describing run folder: config, metrics, predictions, models.

    ``outputs`` maps file names to DataFrames (→ CSV) or dicts (→ JSON).
    ``models`` maps names to fitted forecasters saved in their native format.
    """
    run_dir = ARTIFACTS_DIR / ticker / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    for name, obj in outputs.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(run_dir / f"{name}.csv")
        else:
            with open(run_dir / f"{name}.json", "w") as fh:
                json.dump(obj, fh, indent=2, default=str)
    for name, model in (models or {}).items():
        if hasattr(model, "model_") and hasattr(model.model_, "save_model"):     # XGBoost
            model.model_.save_model(run_dir / f"{name}.json")
        elif hasattr(model, "model_") and hasattr(model.model_, "save"):          # Keras
            model.model_.save(run_dir / f"{name}.keras")
        else:
            import joblib
            joblib.dump(model, run_dir / f"{name}.joblib")
    meta = {"ticker": ticker, "created": datetime.now().isoformat(timespec="seconds"),
            "stocksense_version": __version__, "git_sha": _git_sha(),
            "python": platform.python_version()}
    with open(run_dir / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return run_dir


def log_to_mlflow(ticker: str, cfg: Config, leaderboard: pd.DataFrame,
                  extra_metrics: dict[str, float], run_dir: Path) -> str | None:
    """Log params, per-model metrics and the run folder to MLflow."""
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("stocksense")
    with mlflow.start_run(run_name=f"{ticker}-{run_dir.name}") as run:
        flat = pd.json_normalize(cfg.to_dict(), sep=".").iloc[0].to_dict()
        mlflow.log_params({k: str(v) for k, v in flat.items()} | {"ticker": ticker})
        for model, row in leaderboard.iterrows():
            for metric in ("rmse_price", "mape", "rmse_ret", "dir_acc", "r2_oos", "dm_p"):
                if pd.notna(row.get(metric)):
                    mlflow.log_metric(f"{model}.{metric}", float(row[metric]))
        mlflow.log_metrics({k: float(v) for k, v in extra_metrics.items() if pd.notna(v)})
        mlflow.log_artifacts(str(run_dir))
        return run.info.run_id
