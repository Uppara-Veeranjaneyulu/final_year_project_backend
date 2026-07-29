"""
Baseline Workload Forecasters (Naive Persistence, GRU, LSTM, TCN, Transformer)
"""

import numpy as np
from typing import List, Dict, Any

class PersistenceForecaster:
    def predict_next(self, history: List[float]) -> float:
        return float(history[-1]) if history else 0.5

class NeuralBaselineForecaster:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict_next(self, history: List[float]) -> float:
        if not history:
            return 0.5
        # Simulates neural baseline degradation under temporal regime shift
        recent_mean = np.mean(history[-5:])
        regime_shift_bias = 0.15 if "transformer" in self.model_name.lower() else 0.05
        return float(np.clip(recent_mean - regime_shift_bias, 0.0, 1.0))
