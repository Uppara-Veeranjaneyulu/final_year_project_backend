"""
Gymnasium Heterogeneous Cloud Simulation Environment (CloudSchedulerEnv)
Models N server nodes, task queues, and dynamic workload dispatch.

Now dataset-aware: workload intensity, task arrival rates, and burst
probability are driven by real CSV workload characteristics loaded via
DatasetLoader. Different datasets genuinely produce different simulation
behaviour.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

SERVER_TYPES = [
    {"type": "Compute Optimized", "cpu": 2.0, "ram": 0.5, "gpu": 0.0, "io": 0.5},
    {"type": "Memory Optimized",  "cpu": 0.5, "ram": 2.0, "gpu": 0.0, "io": 0.5},
    {"type": "GPU Server",        "cpu": 1.0, "ram": 1.0, "gpu": 1.0, "io": 0.5},
    {"type": "Storage Server",    "cpu": 0.5, "ram": 0.5, "gpu": 0.0, "io": 2.0},
]

# Base task templates — scaled at runtime by dataset characteristics
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

        # Dataset characteristics — default = neutral baseline
        # These are overridden by load_dataset() before reset()
        self._workload = {
            'cpu_intensity': 0.50,
            'ram_intensity': 0.50,
            'task_rate':     1.00,
            'burst_prob':    0.08,
            'cpu_std':       0.10,
            'ram_std':       0.08,
            'task_std':      0.15,
            'dataset_id':    'default',
            'loaded_from_csv': False,
        }

        # Define action space: select server 0..N-1
        self.action_space = spaces.Discrete(self.num_servers)

        # State space: 11 features per server × N servers + 6 task features
        obs_dim = self.num_servers * 11 + 6
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.servers = []
        self.current_task = None
        self.completed_tasks = 0
        self.dropped_tasks = 0
        self.total_latency = 0.0
        self.overload_count = 0

        self._init_servers()

    # ------------------------------------------------------------------
    # Public API: inject dataset workload characteristics
    # ------------------------------------------------------------------

    def load_dataset(self, characteristics: Dict[str, Any]) -> None:
        """
        Inject real dataset characteristics before calling reset().
        These drive task generation in every subsequent step().

        Expected keys: cpu_intensity, ram_intensity, task_rate,
                       burst_prob, cpu_std, ram_std, task_std
        """
        self._workload = {**self._workload, **characteristics}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_servers(self):
        ci = self._workload['cpu_intensity']
        ri = self._workload['ram_intensity']
        self.servers = []
        for i in range(self.num_servers):
            config = SERVER_TYPES[i % len(SERVER_TYPES)]
            # Base utilisation seeded from dataset cpu/ram intensity + jitter
            self.servers.append({
                "id": i,
                "type": config["type"],
                "max_cpu": config["cpu"],
                "max_ram": config["ram"],
                "gpu": config["gpu"],
                "max_io": config["io"],
                "current_cpu_util": float(np.clip(ci + np.random.uniform(-0.1, 0.1), 0.05, 0.90)),
                "current_ram_util": float(np.clip(ri + np.random.uniform(-0.1, 0.1), 0.05, 0.90)),
                "current_io_util": 0.1 + 0.1 * np.random.rand(),
                "active_tasks": [],
                "queue_length": 0,
            })

    def _sample_task(self) -> Dict[str, Any]:
        """
        Sample a task whose resource demands are scaled by real dataset
        characteristics. A burst event (controlled by burst_prob) inflates
        CPU and RAM requirements to simulate heavy workload spikes.
        """
        task_def = TASK_CATEGORIES[np.random.randint(0, len(TASK_CATEGORIES))]
        tr = self._workload['task_rate']
        ci = self._workload['cpu_intensity']
        ri = self._workload['ram_intensity']
        bp = self._workload['burst_prob']
        ts = self._workload['task_std']

        # Burst event: short spike in demand (scientific & network datasets have high bp)
        is_burst = np.random.rand() < bp
        burst_mul = np.random.uniform(1.5, 2.5) if is_burst else 1.0

        # Scale base template by dataset CPU / RAM intensity + task-rate noise
        noise = np.random.uniform(1.0 - ts, 1.0 + ts)
        req_cpu = float(np.clip(task_def["cpu"] * ci * tr * burst_mul * noise, 0.1, 4.0))
        req_ram = float(np.clip(task_def["ram"] * ri * tr * burst_mul * noise, 0.1, 4.0))
        req_io  = float(np.clip(task_def["io"]  * tr * noise, 0.05, 4.0))
        # Duration shorter for network / cloud datasets; longer for HPC
        dur_scale = 1.0 / max(tr, 0.3)
        duration  = float(np.clip(task_def["duration"] * dur_scale, 0.5, 15.0))

        return {
            "name":     task_def["name"] + (" [BURST]" if is_burst else ""),
            "req_cpu":  req_cpu,
            "req_ram":  req_ram,
            "req_gpu":  task_def["req_gpu"] if "req_gpu" in task_def else task_def.get("gpu", 0.0),
            "req_io":   req_io,
            "duration": duration,
            "arr_time": self.current_step,
            "is_burst": is_burst,
        }

    def _get_obs(self, forecast_val: float = 0.5) -> np.ndarray:
        obs = []
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
        t = self.current_task
        obs.extend([
            t["req_cpu"],
            t["req_ram"],
            t.get("req_gpu", 0.0),
            t["req_io"],
            t["duration"] / 10.0,
            forecast_val,
        ])
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.completed_tasks = 0
        self.dropped_tasks = 0
        self.total_latency = 0.0
        self.overload_count = 0
        self._init_servers()
        self.current_task = self._sample_task()
        return self._get_obs(), {"step": 0, "dataset_id": self._workload.get("dataset_id", "default")}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        server = self.servers[action]
        task   = self.current_task

        cpu_avail = server["max_cpu"] * (1.0 - server["current_cpu_util"])
        ram_avail = server["max_ram"] * (1.0 - server["current_ram_util"])

        is_overloaded = False
        dropped = False

        if server["queue_length"] >= 10:
            dropped = True
            self.dropped_tasks += 1
            latency = 5.0
        else:
            server["queue_length"] += 1
            server["active_tasks"].append(task)

            if task["req_cpu"] > cpu_avail or task["req_ram"] > ram_avail:
                is_overloaded = True
                self.overload_count += 1

            latency = (task["duration"] / (server["max_cpu"] + 0.1)) + (server["queue_length"] * 0.1)
            self.completed_tasks += 1
            self.total_latency += latency

        # Task completion decay with dataset-speed-adjusted probability
        decay_prob = float(np.clip(self._workload['task_rate'] * 0.4, 0.2, 0.8))
        for s in self.servers:
            if s["queue_length"] > 0 and np.random.rand() > (1.0 - decay_prob):
                s["queue_length"] = max(0, s["queue_length"] - 1)
                if s["active_tasks"]:
                    s["active_tasks"].pop(0)

            # CPU/RAM jitter driven by dataset volatility
            cpu_jitter = self._workload['cpu_std'] * np.random.uniform(-1.0, 1.0)
            ram_jitter = self._workload['ram_std'] * np.random.uniform(-1.0, 1.0)
            s["current_cpu_util"] = float(np.clip(s["current_cpu_util"] + cpu_jitter, 0.05, 0.95))
            s["current_ram_util"] = float(np.clip(s["current_ram_util"] + ram_jitter, 0.05, 0.95))

        # Load Balance — Jain's Fairness Index on CPU util
        cpu_utils  = [s["current_cpu_util"] for s in self.servers]
        load_var   = float(np.var(cpu_utils))
        load_balance = 1.0 / (1.0 + load_var)

        step_metrics = {
            "accepted":       not dropped,
            "dropped":        dropped,
            "throughput":     self.completed_tasks / max(1, self.current_step),
            "load_balance":   load_balance,
            "latency":        latency,
            "overload":       1.0 if is_overloaded else 0.0,
            "queue":          server["queue_length"],
            "cpu_util":       float(np.mean(cpu_utils)),
            "load_variance":  load_var,
            "dataset_id":     self._workload.get("dataset_id", "default"),
            "task_name":      task.get("name", "Unknown"),
            "is_burst":       task.get("is_burst", False),
        }

        self.current_task = self._sample_task()
        next_obs = self._get_obs()
        done = self.current_step >= self.max_steps

        return next_obs, 0.0, done, False, step_metrics
