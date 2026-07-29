from flask import Blueprint, jsonify, request
from app.routes.simulation_routes import env_instance, reward_manager
from app.rl.schedulers import get_scheduler

scheduler_bp = Blueprint('scheduler', __name__)

@scheduler_bp.route('/schedule', methods=['POST'])
def schedule_task():
    data = request.get_json() or {}
    policy_name = data.get('policy', 'ppo')
    
    scheduler = get_scheduler(policy_name, num_servers=env_instance.num_servers)
    obs = env_instance._get_obs()
    
    selected_action = scheduler.select_server(obs, num_servers=env_instance.num_servers)
    next_obs, _, done, _, metrics = env_instance.step(selected_action)
    
    reward = reward_manager.compute_reward(metrics)
    
    return jsonify({
        "policy": policy_name,
        "selected_server": selected_action,
        "reward": reward,
        "done": done,
        "metrics": metrics,
        "current_weights": reward_manager.current_weights
    }), 200

@scheduler_bp.route('/metrics/reward-weights', methods=['GET'])
def get_reward_weights_trajectory():
    return jsonify({
        "current_weights": reward_manager.current_weights,
        "history": reward_manager.history,
        "targets": reward_manager.targets,
        "adaptation_rate": reward_manager.eta
    }), 200
