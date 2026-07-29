"""
Backend Test Script to verify Gymnasium environment, AdaptiveRewardManager, and Forecaster modules
"""

import sys
import io

# Ensure UTF-8 output encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_tests():
    print("=== Testing Cloud Scheduler Backend Modules ===")
    
    # 1. Test AdaptiveRewardManager
    from app.rl.adaptive_reward import AdaptiveRewardManager
    arm = AdaptiveRewardManager()
    step_reward = arm.compute_reward({
        "acceptance": 1.0,
        "throughput": 0.5,
        "load_balance": 0.8,
        "latency": 0.15,
        "drop_rate": 0.02,
        "overload": 0.0,
        "queue": 2.0
    })
    print(f"[OK] AdaptiveRewardManager initial step reward: {step_reward:.4f}")
    
    updated_weights = arm.update_at_episode_end({
        "acceptance": 0.90,  # Below target 0.95 -> violation
        "throughput": 0.50,  # Below target 0.65 -> violation
        "load_balance": 0.85,
        "latency": 0.25,      # Above target 0.20 -> violation
        "drop_rate": 0.06,    # Above target 0.05 -> violation
        "overload": 1.0,
        "queue": 6.0
    })
    print(f"[OK] AdaptiveRewardManager updated weights after episode violation: {updated_weights}")
    
    # 2. Test Gymnasium CloudSchedulerEnv
    from app.rl.environment import CloudSchedulerEnv
    env = CloudSchedulerEnv(num_servers=4)
    obs, info = env.reset()
    assert obs.shape == (50,), f"Expected shape (50,), got {obs.shape}"
    print(f"[OK] CloudSchedulerEnv reset successful, state space shape: {obs.shape}")
    
    next_obs, reward, done, truncated, metrics = env.step(action=0)
    print(f"[OK] CloudSchedulerEnv step(action=0) metrics: throughput={metrics['throughput']:.3f}, latency={metrics['latency']:.3f}")
    
    # 3. Test SARIMAX Forecaster
    from app.forecasting.sarimax_model import SARIMAXForecaster
    forecaster = SARIMAXForecaster()
    history = [0.3, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55, 0.5]
    pred = forecaster.predict_next(history)
    print(f"[OK] SARIMAXForecaster 1-step prediction: {pred:.4f}")
    
    print("\nALL BACKEND MODULE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
