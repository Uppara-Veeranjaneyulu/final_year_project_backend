from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List

from app.api.v1.endpoints.simulation import env_instance, reward_manager
from app.rl.schedulers import get_scheduler

router = APIRouter()

class ScheduleRequest(BaseModel):
    policy: str = "ppo"

@router.post("/schedule")
def schedule_task(req: ScheduleRequest):
    scheduler = get_scheduler(req.policy, num_servers=env_instance.num_servers)
    obs = env_instance._get_obs()
    
    selected_action = scheduler.select_server(obs, num_servers=env_instance.num_servers)
    next_obs, _, done, _, metrics = env_instance.step(selected_action)
    
    reward = reward_manager.compute_reward(metrics)
    
    return {
        "policy": req.policy,
        "selected_server": selected_action,
        "reward": reward,
        "done": done,
        "metrics": metrics,
        "current_weights": reward_manager.current_weights
    }

@router.get("/metrics/reward-weights")
def get_reward_weights_trajectory():
    return {
        "current_weights": reward_manager.current_weights,
        "history": reward_manager.history,
        "targets": reward_manager.targets,
        "adaptation_rate": reward_manager.eta
    }
