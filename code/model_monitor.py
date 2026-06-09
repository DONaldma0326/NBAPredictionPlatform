from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
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
DRIFT_PSI_THRESHOLD = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.2"))
LOG_LOSS_DELTA_THRESHOLD = float(os.environ.get("LOG_LOSS_DELTA_THRESHOLD", "0.03"))
ROC_AUC_DROP_THRESHOLD = float(os.environ.get("ROC_AUC_DROP_THRESHOLD", "0.03"))
MONITOR_DAYS = int(os.environ.get("MONITOR_DAYS", "30"))


@dataclass
class HealthReport:
    should_retrain: bool
    reasons: list[str]
    drift_by_feature: dict[str, float]
    average_psi: float
    recent_metrics: dict[str, float] | None
    baseline_metrics: dict[str, float] | None
    recent_rows: int
    reference_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "drift_by_feature": self.drift_by_feature,
            "average_psi": self.average_psi,
            "recent_metrics": self.recent_metrics,
            "baseline_metrics": self.baseline_metrics,
            "recent_rows": self.recent_rows,
            "reference_rows": self.reference_rows,
        }


def load_history_frame() -> pd.DataFrame:
    sql = f"SELECT * FROM {ATHENA_DB}.{TRAINING_TABLE}"
    frame = read_athena_query(sql)
    if frame.empty:
        raise ValueError(f"No rows found in {ATHENA_DB}.{TRAINING_TABLE}.")
    missing = sorted(set(FEATURE_COLS + [LABEL_COL, DATE_COL]) - set(frame.columns))
    if missing:
        raise ValueError(f"Training table missing columns: {', '.join(missing)}")
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="coerce")
    frame = frame.dropna(subset=[DATE_COL]).sort_values([DATE_COL]).reset_index(drop=True)
    return frame


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 5 or len(cur) < 5:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(ref))
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / len(cur))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _get_baseline_metrics() -> dict[str, float] | None:
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
    normalized = normalize_metric_dict(runs[0].data.metrics)
    out: dict[str, float] = {}
    for key in ("accuracy", "log_loss", "roc_auc"):
        value = normalized.get(key)
        if value is not None:
            out[key] = float(value)
    return out or None


def _load_current_model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def _score_frame(model: XGBClassifier, frame: pd.DataFrame) -> dict[str, float]:
    features = frame[FEATURE_COLS]
    labels = frame[LABEL_COL].astype(int)
    proba = model.predict_proba(features)[:, 1]
    preds = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "log_loss": float(log_loss(labels, proba)),
        "roc_auc": float(roc_auc_score(labels, proba)),
    }


def assess_model_health(days: int = MONITOR_DAYS) -> HealthReport:
    history = load_history_frame()
    cutoff = history[DATE_COL].max() - pd.Timedelta(days=days)
    reference = history[history[DATE_COL] < cutoff].copy()
    recent = history[history[DATE_COL] >= cutoff].copy()
    if reference.empty or recent.empty:
        return HealthReport(
            should_retrain=False,
            reasons=["insufficient_recent_data"],
            drift_by_feature={},
            average_psi=0.0,
            recent_metrics=None,
            baseline_metrics=_get_baseline_metrics(),
            recent_rows=int(len(recent)),
            reference_rows=int(len(reference)),
        )

    drift_by_feature = {feature: _psi(reference[feature], recent[feature]) for feature in FEATURE_COLS}
    average_psi = float(np.mean(list(drift_by_feature.values()))) if drift_by_feature else 0.0

    model = _load_current_model()
    recent_metrics = _score_frame(model, recent)
    baseline_metrics = _get_baseline_metrics()

    reasons: list[str] = []
    if average_psi >= DRIFT_PSI_THRESHOLD:
        reasons.append(f"feature_drift_average_psi>{DRIFT_PSI_THRESHOLD}")
    if baseline_metrics is not None:
        baseline_log_loss = baseline_metrics.get("log_loss")
        baseline_auc = baseline_metrics.get("roc_auc")
        if baseline_log_loss is not None and recent_metrics["log_loss"] > baseline_log_loss + LOG_LOSS_DELTA_THRESHOLD:
            reasons.append("log_loss_degraded")
        if baseline_auc is not None and recent_metrics["roc_auc"] < baseline_auc - ROC_AUC_DROP_THRESHOLD:
            reasons.append("roc_auc_degraded")

    return HealthReport(
        should_retrain=bool(reasons),
        reasons=reasons,
        drift_by_feature=drift_by_feature,
        average_psi=average_psi,
        recent_metrics=recent_metrics,
        baseline_metrics=baseline_metrics,
        recent_rows=int(len(recent)),
        reference_rows=int(len(reference)),
    )
