"""
Adaptive Dynamic Reward System (AdaptiveRewardManager)
Automatically re-weights multi-objective reward coefficients based on normalized violations of target objectives.
"""

import numpy as np
from typing import Dict, List, Any

class AdaptiveRewardManager:
    def __init__(
        self,
        learning_rate: float = 0.05,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        initial_weights: Dict[str, float] = None,
        targets: Dict[str, float] = None
    ):
        self.eta = learning_rate
        self.w_min = min_weight
        self.w_max = max_weight
        
        # Initial baseline weight vector w^0
        self.initial_weights = initial_weights or {
            'acceptance': 1.0,
            'throughput': 1.0,
            'load_balance': 1.0,
            'latency': 1.0,
            'drop_rate': 1.0,
            'overload': 1.0,
            'queue': 1.0
        }
        
        self.current_weights = self.initial_weights.copy()
        
        # Target SLA vector g
        self.targets = targets or {
            'acceptance': 0.95,
            'throughput': 0.65,
            'load_balance': 0.90,  # Maximize load balance index
            'latency': 0.20,       # Max latency 0.20s
            'drop_rate': 0.05,     # Max drop rate 5%
            'overload': 0.0,       # Zero overload events
            'queue': 5.0           # Max avg queue length 5
        }
        
        # History trajectory logs
        self.history: List[Dict[str, float]] = [self.current_weights.copy()]
        self.violations_history: List[Dict[str, float]] = []

    def compute_reward(self, step_metrics: Dict[str, float]) -> float:
        """
        Compute step reward R_k using current dynamic weight vector w(t):
        R = w1*A + w2*T + w3*B - w4*L - w5*D - w6*O - w7*Q
        """
        w = self.current_weights
        
        a = step_metrics.get('acceptance', 1.0 if step_metrics.get('accepted', True) else 0.0)
        t = step_metrics.get('throughput', 0.0)
        b = step_metrics.get('load_balance', 0.0)
        l = step_metrics.get('latency', 0.0)
        d = step_metrics.get('drop_rate', 1.0 if step_metrics.get('dropped', False) else 0.0)
        o = step_metrics.get('overload', 1.0 if step_metrics.get('is_overloaded', False) else 0.0)
        q = step_metrics.get('queue', 0.0)
        
        reward = (
            w['acceptance'] * a +
            w['throughput'] * t +
            w['load_balance'] * b -
            w['latency'] * l -
            w['drop_rate'] * d -
            w['overload'] * o -
            w['queue'] * q
        )
        return float(reward)

    def update_at_episode_end(self, episode_summary: Dict[str, float]) -> Dict[str, float]:
        """
        Executes end-of-episode weight recalibration based on observed violations e_{i,t}
        """
        errors = {}
        g = self.targets
        
        # 1. Compute normalized errors
        # Maximization objectives
        for m in ['acceptance', 'throughput', 'load_balance']:
            obs = episode_summary.get(m, 0.0)
            target = g[m]
            errors[m] = (target - obs) / target if target > 0 else 0.0
            
        # Minimization objectives
        for m in ['latency', 'queue']:
            obs = episode_summary.get(m, 0.0)
            target = g[m]
            errors[m] = (obs - target) / target if target > 0 else obs
            
        # Zero-target objectives
        for m in ['drop_rate', 'overload']:
            errors[m] = episode_summary.get(m, 0.0)

        # 2. Priority allocation p_{i,t}
        positive_errors = {m: max(err, 0.0) for m, err in errors.items()}
        total_violation = sum(positive_errors.values())
        
        priorities = {}
        if total_violation > 0:
            priorities = {m: positive_errors[m] / total_violation for m in errors}
        else:
            priorities = {m: 0.0 for m in errors}
            
        # 3. Update weights
        new_weights = {}
        for m, w_val in self.current_weights.items():
            err = errors[m]
            p = priorities[m]
            if err > 0:
                # Violation present -> increase weight
                updated = w_val + self.eta * p * err
            else:
                # Objective satisfied -> decay toward initial weight w^0
                updated = w_val - self.eta * (w_val - self.initial_weights[m])
                
            # 4. Clamp to bounds
            clamped = float(np.clip(updated, self.w_min, self.w_max))
            new_weights[m] = clamped
            
        self.current_weights = new_weights
        self.history.append(new_weights.copy())
        self.violations_history.append(errors.copy())
        
        return self.current_weights

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_weights": self.current_weights,
            "adaptation_rate": self.eta,
            "bounds": [self.w_min, self.w_max],
            "episode_count": len(self.history) - 1,
        }
