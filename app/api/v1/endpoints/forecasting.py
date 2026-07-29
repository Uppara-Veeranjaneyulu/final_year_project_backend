from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

from app.forecasting.sarimax_model import SARIMAXForecaster
from app.forecasting.baselines import PersistenceForecaster, NeuralBaselineForecaster

router = APIRouter()

sarimax = SARIMAXForecaster()
persistence = PersistenceForecaster()

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
        pred = neural.predict_next(req.history)
        
    return {
        "model": req.model,
        "predicted_workload": pred,
        "input_window_length": len(req.history)
    }

@router.get("/evaluation")
def get_forecasting_models_results():
    # Return paper Table 1 verified forecasting results
    return {
        "results": [
            { "model": "SARIMAX", "mae": 0.0248, "rmse": 0.0708, "r2": 0.270, "selected": True },
            { "model": "Naive Persistence", "mae": 0.0196, "rmse": 0.0789, "r2": 0.092, "selected": False },
            { "model": "GRU", "mae": 0.2000, "rmse": 0.2352, "r2": -7.06, "selected": False },
            { "model": "LSTM", "mae": 0.4806, "rmse": 0.4981, "r2": -35.15, "selected": False },
            { "model": "TCN", "mae": 0.7047, "rmse": 0.7419, "r2": -79.19, "selected": False },
            { "model": "Bidirectional LSTM", "mae": 0.7683, "rmse": 0.8043, "r2": -93.25, "selected": False },
            { "model": "Transformer Encoder", "mae": 2.0538, "rmse": 2.1734, "r2": -687.23, "selected": False }
        ]
    }
