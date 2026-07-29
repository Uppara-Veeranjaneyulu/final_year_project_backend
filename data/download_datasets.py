"""
Dataset Downloader & Synthesizer Helper Script
Generates processed time-series sample files for offline development and testing.
"""

import os
import sys
import io
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASETS_CONFIG = {
    "google-cluster-v1": {"source": "Google Inc. (2011)", "type": "cloud"},
    "google-cluster-v2": {"source": "Google Inc. (2019)", "type": "cloud"},
    "bitbrains": {"source": "Bitbrains IT Services", "type": "vm"},
    "azure-public": {"source": "Microsoft Azure (2017)", "type": "vm"},
    "alibaba-cluster": {"source": "Alibaba Group (2018)", "type": "microservices"},
    "spitzer": {"source": "NASA IRSA", "type": "scientific"},
    "xmm-newton": {"source": "ESA XSA Archive", "type": "scientific"},
    "parallel-workloads": {"source": "HUJI PWA", "type": "hpc"},
    "hpc2n": {"source": "HPC2N Center", "type": "hpc"},
    "caida-2025": {"source": "CAIDA Passive 100G", "type": "network"},
}

def generate_sample_traces():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Generating processed sample workload traces in: {processed_dir}\n")
    
    for ds_id, cfg in DATASETS_CONFIG.items():
        out_path = os.path.join(processed_dir, f"{ds_id}_5min.csv")
        
        np.random.seed(hash(ds_id) % 10000)
        n_steps = 288  # 24 hours at 5-minute sampling resolution
        t = np.linspace(0, 24, n_steps)
        
        # Diurnal workload curve + noise
        cpu_util = 0.4 + 0.3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.05, n_steps)
        ram_util = 0.5 + 0.2 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.04, n_steps)
        
        if cfg["type"] == "scientific":
            # Add bursty observation spikes
            spikes = (np.random.rand(n_steps) > 0.93) * 0.35
            cpu_util += spikes
            
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="5min"),
            "cpu_utilization": np.clip(cpu_util, 0.0, 1.0),
            "ram_utilization": np.clip(ram_util, 0.0, 1.0),
            "task_count": np.random.poisson(lam=45, size=n_steps),
        })
        
        df.to_csv(out_path, index=False)
        print(f"  [OK] Saved {ds_id} trace: {n_steps} timesteps -> {out_path}")
        
    print("\nALL SAMPLE WORKLOAD TRACES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    generate_sample_traces()
