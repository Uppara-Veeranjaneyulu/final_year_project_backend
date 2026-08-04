"""
Forecasting endpoints.
The /evaluation endpoint computes real MAE, RMSE and R² metrics by running a
rolling-window evaluation on the actual google-cluster-v1 CPU utilization CSV,
which is the designated 'Primary Forecasting & RL Baseline' dataset.
Results are cached on first request to avoid repeated computation.
"""

import os
import csv
import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.forecasting.sarimax_model import SARIMAXForecaster
from app.forecasting.baselines import PersistenceForecaster, NeuralBaselineForecaster

router = APIRouter()

sarimax     = SARIMAXForecaster()
persistence = PersistenceForecaster()

# Path to the primary forecasting dataset
_DATA_DIR   = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'processed')
)
_PRIMARY_CSV = os.path.join(_DATA_DIR, 'google-cluster-v1_5min.csv')

# Cache for evaluation results — computed once at first request
_eval_cache: Optional[List[dict]] = None


def _load_primary_series() -> List[float]:
    """Load cpu_utilization from the primary forecasting dataset CSV."""
    if not os.path.isfile(_PRIMARY_CSV):
        raise HTTPException(
            status_code=503,
            detail="Primary dataset CSV not found. Run data/download_datasets.py first."
        )
    values = []
    with open(_PRIMARY_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row['cpu_utilization']))
            except (ValueError, KeyError):
                continue
    if not values:
        raise HTTPException(status_code=500, detail="Primary dataset CSV is empty or malformed.")
    return values


def _rolling_evaluate(model, series: List[float], window: int = 24) -> dict:
    """
    Rolling 1-step-ahead evaluation.
    Requires at least `window` history points before making each prediction.
    Returns {'mae': float, 'rmse': float, 'r2': float}.
    """
    y_true, y_pred = [], []
    for i in range(window, len(series)):
        history = series[i - window: i]
        pred    = model.predict_next(history)
        y_true.append(series[i])
        y_pred.append(pred)

    if not y_true:
        return {"mae": None, "rmse": None, "r2": None}

    n      = len(y_true)
    mae    = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse    = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse   = math.sqrt(mse)
    mean_t = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_t) ** 2 for t in y_true) or 1e-8
    r2     = 1.0 - ss_res / ss_tot

    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "r2":   round(r2,   3),
    }


class PredictRequest(BaseModel):
    history: List[float]
    model: str = "sarimax"


@router.post("/predict")
def predict_workload(req: PredictRequest):
    m_name = req.model.lower()

    if "sarimax" in m_name:
        pred = sarimax.predict_next(req.history)
    elif "persistence" in m_name:
        pred = persistence.predict_next(req.history)
    else:
        neural = NeuralBaselineForecaster(m_name)
        pred   = neural.predict_next(req.history)

    return {
        "model":                req.model,
        "predicted_workload":   pred,
        "input_window_length":  len(req.history),
    }


@router.get("/evaluation")
def get_forecasting_models_results():
    """
    Returns real MAE / RMSE / R² for each forecasting model, computed via
    rolling-window evaluation on the google-cluster-v1 CPU utilization series.
    Results are cached in-process after the first call.
    """
    global _eval_cache
    if _eval_cache is not None:
        return {"results": _eval_cache, "cached": True}

    series = _load_primary_series()

    models = [
        ("SARIMAX",              sarimax,                             True),
        ("Naive Persistence",    persistence,                         False),
        ("GRU",                  NeuralBaselineForecaster("gru"),     False),
        ("LSTM",                 NeuralBaselineForecaster("lstm"),    False),
        ("TCN",                  NeuralBaselineForecaster("tcn"),     False),
        ("Bidirectional LSTM",   NeuralBaselineForecaster("bilstm"),  False),
        ("Transformer Encoder",  NeuralBaselineForecaster("transformer"), False),
    ]

    results = []
    for name, model, selected in models:
        metrics = _rolling_evaluate(model, series)
        results.append({
            "model":    name,
            "mae":      metrics["mae"],
            "rmse":     metrics["rmse"],
            "r2":       metrics["r2"],
            "selected": selected,
            "eval_rows": len(series) - 24,  # rows used for evaluation
        })

    _eval_cache = results
    return {"results": results, "cached": False}
