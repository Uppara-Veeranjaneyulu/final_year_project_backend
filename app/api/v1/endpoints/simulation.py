from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.rl.environment import CloudSchedulerEnv
from app.rl.adaptive_reward import AdaptiveRewardManager

router = APIRouter()

# Global simulation instance for API access
env_instance = CloudSchedulerEnv(num_servers=4)
reward_manager = AdaptiveRewardManager()

class StepRequest(BaseModel):
    action: int

class ResetResponse(BaseModel):
    observation: List[float]
    status: str

@router.post("/reset", response_model=ResetResponse)
def reset_simulation():
    obs, info = env_instance.reset()
    return {
        "observation": obs.tolist(),
        "status": "reset_success"
    }

@router.post("/step")
def step_simulation(req: StepRequest):
    if req.action < 0 or req.action >= env_instance.num_servers:
        raise HTTPException(status_code=400, detail="Invalid action server index")
        
    next_obs, _, done, _, step_metrics = env_instance.step(req.action)
    
    # Compute adaptive reward
    reward = reward_manager.compute_reward(step_metrics)
    
    # Check if episode ended
    if done:
        updated_weights = reward_manager.update_at_episode_end({
            "acceptance": 0.92,
            "throughput": step_metrics["throughput"],
            "load_balance": step_metrics["load_balance"],
            "latency": step_metrics["latency"],
            "drop_rate": 0.04,
            "overload": step_metrics["overload"],
            "queue": step_metrics["queue"],
        })
    else:
        updated_weights = reward_manager.current_weights

    return {
        "observation": next_obs.tolist(),
        "reward": reward,
        "done": done,
        "metrics": step_metrics,
        "reward_weights": updated_weights
    }

@router.get("/status")
def get_simulation_status():
    return {
        "current_step": env_instance.current_step,
        "max_steps": env_instance.max_steps,
        "num_servers": env_instance.num_servers,
        "completed_tasks": env_instance.completed_tasks,
        "dropped_tasks": env_instance.dropped_tasks,
        "overload_count": env_instance.overload_count,
        "adaptive_reward_status": reward_manager.get_status()
    }
