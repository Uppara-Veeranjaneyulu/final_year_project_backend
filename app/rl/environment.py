"""
Gymnasium Heterogeneous Cloud Simulation Environment (CloudSchedulerEnv)
Models N server nodes, task queues, and dynamic workload dispatch.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List

SERVER_TYPES = [
    {"type": "Compute Optimized", "cpu": 2.0, "ram": 0.5, "gpu": 0.0, "io": 0.5},
    {"type": "Memory Optimized",  "cpu": 0.5, "ram": 2.0, "gpu": 0.0, "io": 0.5},
    {"type": "GPU Server",        "cpu": 1.0, "ram": 1.0, "gpu": 1.0, "io": 0.5},
    {"type": "Storage Server",    "cpu": 0.5, "ram": 0.5, "gpu": 0.0, "io": 2.0},
]

TASK_CATEGORIES = [
    {"name": "Video Encoding", "cpu": 1.5, "ram": 0.5, "gpu": 0.0, "io": 0.5, "duration": 5.0},
    {"name": "AI Inference",   "cpu": 0.5, "ram": 1.0, "gpu": 1.0, "io": 0.2, "duration": 3.0},
    {"name": "SQL Analytics",  "cpu": 0.5, "ram": 1.5, "gpu": 0.0, "io": 1.0, "duration": 4.0},
    {"name": "File Backup",    "cpu": 0.2, "ram": 0.2, "gpu": 0.0, "io": 1.8, "duration": 6.0},
    {"name": "Web API Request","cpu": 0.4, "ram": 0.3, "gpu": 0.0, "io": 0.4, "duration": 1.0},
]

class CloudSchedulerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_servers: int = 4, max_steps: int = 300):
        super().__init__()
        self.num_servers = num_servers
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space: select server 0..N-1
        self.action_space = spaces.Discrete(self.num_servers)
        
        # State space: 11 features per server * 4 servers + 6 task features = 50 features total
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(50,), dtype=np.float32
        )
        
        self.servers = []
        self.current_task = None
        self.completed_tasks = 0
        self.dropped_tasks = 0
        self.total_latency = 0.0
        self.overload_count = 0
        
        self._init_servers()

    def _init_servers(self):
        self.servers = []
        for i in range(self.num_servers):
            config = SERVER_TYPES[i % len(SERVER_TYPES)]
            self.servers.append({
                "id": i,
                "type": config["type"],
                "max_cpu": config["cpu"],
                "max_ram": config["ram"],
                "gpu": config["gpu"],
                "max_io": config["io"],
                "current_cpu_util": 0.1 + 0.1 * np.random.rand(),
                "current_ram_util": 0.1 + 0.1 * np.random.rand(),
                "current_io_util": 0.1 + 0.1 * np.random.rand(),
                "active_tasks": [],
                "queue_length": 0,
            })

    def _sample_task(self) -> Dict[str, Any]:
        task_def = TASK_CATEGORIES[np.random.randint(0, len(TASK_CATEGORIES))]
        return {
            "name": task_def["name"],
            "req_cpu": task_def["cpu"],
            "req_ram": task_def["ram"],
            "req_gpu": task_def["gpu"],
            "req_io": task_def["io"],
            "duration": task_def["duration"],
            "arr_time": self.current_step,
        }

    def _get_obs(self, forecast_val: float = 0.5) -> np.ndarray:
        obs = []
        # Server features (11 per server x 4 = 44)
        for s in self.servers:
            obs.extend([
                s["current_cpu_util"],
                s["current_ram_util"],
                s["current_io_util"],
                s["gpu"],
                s["queue_length"] / 10.0,
                s["max_cpu"],
                s["max_ram"],
                s["max_io"],
                len(s["active_tasks"]) / 5.0,
                1.0 if s["current_cpu_util"] > 0.85 else 0.0,
                1.0 if s["current_ram_util"] > 0.85 else 0.0,
            ])
            
        # Current Task features (5 features) + Forecast Feature (1 feature) = 6 features
        t = self.current_task
        obs.extend([
            t["req_cpu"],
            t["req_ram"],
            t["req_gpu"],
            t["req_io"],
            t["duration"] / 10.0,
            forecast_val,
        ])
        
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.completed_tasks = 0
        self.dropped_tasks = 0
        self.total_latency = 0.0
        self.overload_count = 0
        self._init_servers()
        self.current_task = self._sample_task()
        
        return self._get_obs(), {"step": 0}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        server = self.servers[action]
        task = self.current_task
        
        # Check node overload capacity
        cpu_avail = server["max_cpu"] * (1.0 - server["current_cpu_util"])
        ram_avail = server["max_ram"] * (1.0 - server["current_ram_util"])
        
        is_overloaded = False
        dropped = False
        
        if server["queue_length"] >= 10:
            # Queue overflow drop
            dropped = True
            self.dropped_tasks += 1
            latency = 5.0
        else:
            # Assign task
            server["queue_length"] += 1
            server["active_tasks"].append(task)
            
            if task["req_cpu"] > cpu_avail or task["req_ram"] > ram_avail:
                is_overloaded = True
                self.overload_count += 1
                
            latency = (task["duration"] / (server["max_cpu"] + 0.1)) + (server["queue_length"] * 0.1)
            self.completed_tasks += 1
            self.total_latency += latency

        # Update server loads & decay completed tasks
        for s in self.servers:
            if s["queue_length"] > 0 and np.random.rand() > 0.4:
                s["queue_length"] = max(0, s["queue_length"] - 1)
                if s["active_tasks"]:
                    s["active_tasks"].pop(0)
            
            # Dynamic utilization jitter
            s["current_cpu_util"] = np.clip(s["current_cpu_util"] + np.random.uniform(-0.05, 0.05), 0.1, 0.95)
            s["current_ram_util"] = np.clip(s["current_ram_util"] + np.random.uniform(-0.05, 0.05), 0.1, 0.95)

        # Compute Load Balance index (Jain's Fairness Index on CPU util)
        cpu_utils = [s["current_cpu_util"] for s in self.servers]
        load_var = float(np.var(cpu_utils))
        load_balance = 1.0 / (1.0 + load_var)

        step_metrics = {
            "accepted": not dropped,
            "dropped": dropped,
            "throughput": self.completed_tasks / max(1, self.current_step),
            "load_balance": load_balance,
            "latency": latency,
            "overload": 1.0 if is_overloaded else 0.0,
            "queue": server["queue_length"],
            "cpu_util": float(np.mean(cpu_utils)),
            "load_variance": load_var,
        }

        # Next state & task
        self.current_task = self._sample_task()
        next_obs = self._get_obs()
        done = self.current_step >= self.max_steps

        return next_obs, 0.0, done, False, step_metrics
