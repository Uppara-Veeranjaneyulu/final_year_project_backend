import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cloud-rl-secret-key-2024')
    PROJECT_NAME = "Cloud RL Task Scheduler & Forecaster Backend (Flask)"
    VERSION = "1.0.0"
    API_PREFIX = "/api/v1"
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]
    
    # Simulation Defaults
    NUM_SERVERS = 4
    SLIDE_WINDOW_SIZE = 24
    EPISODE_MAX_STEPS = 300
    
    # Adaptive Reward Bounds & Rate
    REWARD_ADAPTATION_RATE = 0.05
    MIN_REWARD_WEIGHT = 0.01
    MAX_REWARD_WEIGHT = 10.0
