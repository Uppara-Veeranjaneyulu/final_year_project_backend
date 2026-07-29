from flask import Blueprint, jsonify, request
from app.forecasting.sarimax_model import SARIMAXForecaster
from app.forecasting.baselines import PersistenceForecaster, NeuralBaselineForecaster

forecasting_bp = Blueprint('forecasting', __name__)

sarimax = SARIMAXForecaster()
persistence = PersistenceForecaster()

@forecasting_bp.route('/predict', methods=['POST'])
def predict_workload():
    data = request.get_json() or {}
    history = data.get('history', [0.5, 0.5])
    model_name = data.get('model', 'sarimax').lower()
    
    if "sarimax" in model_name:
        pred = sarimax.predict_next(history)
    elif "persistence" in model_name:
        pred = persistence.predict_next(history)
    else:
        neural = NeuralBaselineForecaster(model_name)
        pred = neural.predict_next(history)
        
    return jsonify({
        "model": model_name,
        "predicted_workload": pred,
        "input_window_length": len(history)
    }), 200

@forecasting_bp.route('/evaluation', methods=['GET'])
def get_forecasting_models_results():
    return jsonify({
        "results": [
            { "model": "SARIMAX", "mae": 0.0248, "rmse": 0.0708, "r2": 0.270, "selected": True },
            { "model": "Naive Persistence", "mae": 0.0196, "rmse": 0.0789, "r2": 0.092, "selected": False },
            { "model": "GRU", "mae": 0.2000, "rmse": 0.2352, "r2": -7.06, "selected": False },
            { "model": "LSTM", "mae": 0.4806, "rmse": 0.4981, "r2": -35.15, "selected": False },
            { "model": "TCN", "mae": 0.7047, "rmse": 0.7419, "r2": -79.19, "selected": False },
            { "model": "Bidirectional LSTM", "mae": 0.7683, "rmse": 0.8043, "r2": -93.25, "selected": False },
            { "model": "Transformer Encoder", "mae": 2.0538, "rmse": 2.1734, "r2": -687.23, "selected": False }
        ]
    }), 200
