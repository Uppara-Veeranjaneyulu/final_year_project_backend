"""
Task Scheduling Policy Engines: PPO, Least Connections, Round Robin, Random
"""

import numpy as np
from typing import Dict, Any, List

class RandomScheduler:
    def select_server(self, obs: np.ndarray, num_servers: int = 4) -> int:
        return int(np.random.randint(0, num_servers))

class RoundRobinScheduler:
    def __init__(self):
        self.counter = 0

    def select_server(self, obs: np.ndarray, num_servers: int = 4) -> int:
        action = self.counter % num_servers
        self.counter += 1
        return action

class LeastConnectionsScheduler:
    def select_server(self, obs: np.ndarray, num_servers: int = 4) -> int:
        # Extract queue lengths from state space observation (feature offset 4, 15, 26, 37)
        queue_lengths = []
        for i in range(num_servers):
            q_idx = i * 11 + 4
            queue_lengths.append(obs[q_idx])
        return int(np.argmin(queue_lengths))

class PPOScheduler:
    def __init__(self, num_servers: int = 4):
        self.num_servers = num_servers
        # Simulated heuristic weights representing trained PPO policy network
        self.weights = np.random.randn(50, num_servers) * 0.1

    def select_server(self, obs: np.ndarray, num_servers: int = 4) -> int:
        logits = np.dot(obs, self.weights)
        # Softmax sampling or argmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return int(np.argmax(probs))

def get_scheduler(policy_name: str, num_servers: int = 4):
    name = policy_name.lower()
    if "ppo" in name:
        return PPOScheduler(num_servers)
    elif "least" in name:
        return LeastConnectionsScheduler()
    elif "round" in name:
        return RoundRobinScheduler()
    else:
        return RandomScheduler()
