from flask import Blueprint, jsonify, request
from app.rl.environment import CloudSchedulerEnv
from app.rl.adaptive_reward import AdaptiveRewardManager
from app.rl.dataset_loader import load_dataset_characteristics

simulation_bp = Blueprint('simulation', __name__)

# Global instances for API state
env_instance = CloudSchedulerEnv(num_servers=4)
reward_manager = AdaptiveRewardManager()

@simulation_bp.route('/reset', methods=['POST'])
def reset_simulation():
    data = request.get_json() or {}
    dataset_id = data.get('dataset_id', 'hpc2n')

    # Load real CSV characteristics and inject into environment
    characteristics = load_dataset_characteristics(dataset_id)
    env_instance.load_dataset(characteristics)

    obs, info = env_instance.reset()
    return jsonify({
        "observation": obs.tolist(),
        "status": "reset_success",
        "dataset_id": dataset_id,
        "workload_characteristics": {
            "cpu_intensity":    round(characteristics['cpu_intensity'], 4),
            "ram_intensity":    round(characteristics['ram_intensity'], 4),
            "task_rate":        round(characteristics['task_rate'], 4),
            "burst_prob":       round(characteristics['burst_prob'], 4),
            "loaded_from_csv":  characteristics['loaded_from_csv'],
            "num_rows":         characteristics.get('num_rows', 0),
        }
    }), 200

@simulation_bp.route('/step', methods=['POST'])
def step_simulation():
    data = request.get_json() or {}
    action = data.get('action', 0)

    if not isinstance(action, int) or action < 0 or action >= env_instance.num_servers:
        return jsonify({"error": "Invalid action server index"}), 400

    next_obs, _, done, _, step_metrics = env_instance.step(action)

    # Compute dynamic reward
    reward = reward_manager.compute_reward(step_metrics)

    # Episode end recalibration
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

    return jsonify({
        "observation": next_obs.tolist(),
        "reward": reward,
        "done": done,
        "metrics": step_metrics,
        "reward_weights": updated_weights
    }), 200

@simulation_bp.route('/status', methods=['GET'])
def get_simulation_status():
    workload = env_instance._workload
    return jsonify({
        "current_step":      env_instance.current_step,
        "max_steps":         env_instance.max_steps,
        "num_servers":       env_instance.num_servers,
        "completed_tasks":   env_instance.completed_tasks,
        "dropped_tasks":     env_instance.dropped_tasks,
        "overload_count":    env_instance.overload_count,
        "active_dataset":    workload.get("dataset_id", "default"),
        "workload": {
            "cpu_intensity":  round(workload.get("cpu_intensity", 0.5), 4),
            "ram_intensity":  round(workload.get("ram_intensity", 0.5), 4),
            "task_rate":      round(workload.get("task_rate", 1.0), 4),
            "burst_prob":     round(workload.get("burst_prob", 0.08), 4),
            "loaded_from_csv": workload.get("loaded_from_csv", False),
        },
        "adaptive_reward_status": reward_manager.get_status()
    }), 200
