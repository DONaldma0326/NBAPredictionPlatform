from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier
from dotenv import load_dotenv
from model_shared import (
    ATHENA_DB,
    FEATURE_COLS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    TRAINING_TABLE,
    normalize_metric_dict,
    read_athena_query,
)
load_dotenv()
LABEL_COL = "WIN"
DATE_COL = "GAME_DATE"
GAME_ID_COL = "GAME_ID"
TEAM_ID_COL = "TEAM_ID"
DEFAULT_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "50"))
DEFAULT_SEED = int(os.environ.get("RANDOM_STATE", "36"))


def load_training_data() -> pd.DataFrame:
    sql = f"SELECT * FROM {ATHENA_DB}.{TRAINING_TABLE}"
    frame = read_athena_query(sql)
    if frame.empty:
        raise ValueError(f"No rows found in {ATHENA_DB}.{TRAINING_TABLE}.")
    missing = sorted(set(FEATURE_COLS + [LABEL_COL, DATE_COL, GAME_ID_COL, TEAM_ID_COL]) - set(frame.columns))
    if missing:
        raise ValueError(f"Training table missing columns: {', '.join(missing)}")
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce")
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    frame = frame.dropna(subset=[DATE_COL]).sort_values([DATE_COL, GAME_ID_COL, TEAM_ID_COL]).reset_index(drop=True)
    return frame


def split_by_game_date(frame: pd.DataFrame, train_fraction: float = 0.7, val_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Cannot split an empty training frame.")
    game_dates = (
        frame.groupby(GAME_ID_COL, as_index=False)[DATE_COL]
        .min()
        .sort_values([DATE_COL, GAME_ID_COL])
        .reset_index(drop=True)
    )
    game_ids = game_dates[GAME_ID_COL].tolist()
    total_games = len(game_ids)
    train_end = max(1, int(total_games * train_fraction))
    val_end = max(train_end + 1, int(total_games * (train_fraction + val_fraction)))
    train_ids = set(game_ids[:train_end])
    val_ids = set(game_ids[train_end:val_end])
    test_ids = set(game_ids[val_end:])
    train_df = frame[frame[GAME_ID_COL].isin(train_ids)].copy()
    val_df = frame[frame[GAME_ID_COL].isin(val_ids)].copy()
    test_df = frame[frame[GAME_ID_COL].isin(test_ids)].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Train/validation/test split produced an empty partition.")
    return train_df, val_df, test_df


def _xgb_params_from_trial(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": seed,
        "tree_method": "hist",
    }


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, seed: int, n_trials: int) -> tuple[dict[str, Any], tuple[float, float], optuna.Study]:
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _xgb_params_from_trial(trial, seed)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, val_proba), roc_auc_score(y_val, val_proba)

    study = optuna.create_study(
        directions=["minimize", "maximize"],
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = min(study.best_trials, key=lambda trial: (trial.values[0], -trial.values[1]))
    return best_trial.params, (float(best_trial.values[0]), float(best_trial.values[1])), study


def evaluate_model(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "log_loss": float(log_loss(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def _get_latest_mlflow_metrics() -> dict[str, float] | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    metrics = normalize_metric_dict(runs[0].data.metrics)
    out: dict[str, float] = {}
    for key in ("accuracy", "log_loss", "roc_auc"):
        value = metrics.get(key)
        if value is not None:
            out[key] = float(value)
    return out or None


def _candidate_is_better(candidate: dict[str, float], baseline: dict[str, float] | None) -> bool:
    if baseline is None:
        return True
    baseline_log_loss = baseline.get("log_loss")
    baseline_auc = baseline.get("roc_auc")
    if baseline_log_loss is None or baseline_auc is None:
        return True
    return candidate["log_loss"] < baseline_log_loss and candidate["roc_auc"] >= baseline_auc


def _save_model(model: XGBClassifier, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(destination))


def train_and_promote_model(triggered_by: str = "manual", n_trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    frame = load_training_data()
    train_df, val_df, test_df = split_by_game_date(frame)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL].astype(int)
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[LABEL_COL].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[LABEL_COL].astype(int)

    best_params, val_objectives, study = tune_hyperparameters(X_train, y_train, X_val, y_val, seed=seed, n_trials=n_trials)
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "random_state": seed,
        "tree_method": "hist",
        **best_params,
    }

    final_model = XGBClassifier(**final_params)
    X_train_full = pd.concat([train_df, val_df], ignore_index=True)[FEATURE_COLS]
    y_train_full = pd.concat([train_df, val_df], ignore_index=True)[LABEL_COL].astype(int)
    final_model.fit(X_train_full, y_train_full, verbose=False)

    test_metrics = evaluate_model(final_model, X_test, y_test)
    baseline_metrics = _get_latest_mlflow_metrics()
    promoted = _candidate_is_better(test_metrics, baseline_metrics)

    model_path = Path(MODEL_PATH)
    candidate_path = model_path.with_name(f"{model_path.stem}_candidate_{triggered_by}.json")
    _save_model(final_model, candidate_path)

    if promoted:
        if model_path.exists():
            backup_path = model_path.with_name(f"{model_path.stem}.backup.json")
            shutil.copy2(model_path, backup_path)
        shutil.copy2(candidate_path, model_path)

    with mlflow.start_run(run_name=f"retrain-{triggered_by}") as run:
        mlflow.log_params({**best_params, "triggered_by": triggered_by, "split_strategy": "game_date"})
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("validation_log_loss", val_objectives[0])
        mlflow.log_metric("validation_roc_auc", val_objectives[1])
        mlflow.set_tags(
            {
                "triggered_by": triggered_by,
                "promoted": str(promoted).lower(),
                "split": "game_date",
            }
        )
        mlflow.xgboost.log_model(final_model, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "promoted": promoted,
            "candidate_path": str(candidate_path),
            "model_path": str(model_path),
            "baseline_metrics": baseline_metrics,
            "validation_metrics": {
                "log_loss": val_objectives[0],
                "roc_auc": val_objectives[1],
            },
            "test_metrics": test_metrics,
            "best_params": best_params,
            "n_trials": n_trials,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "study_best_trials": len(study.best_trials),
        }
