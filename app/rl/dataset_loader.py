"""
DatasetLoader — reads real processed CSV workload traces and extracts
workload characteristics (cpu_intensity, ram_intensity, task_rate, burst_prob)
that drive the CloudSchedulerEnv task generator.

Supported datasets (10 files in data/processed/):
  hpc2n, google-cluster-v1, google-cluster-v2, bitbrains,
  azure-public, alibaba-cluster, spitzer, xmm-newton,
  parallel-workloads, caida-2025
"""

import os
import csv
import numpy as np
from typing import Dict, Any, Optional

# Absolute path to the data/processed directory
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')

# Mapping: dataset_id → CSV file name
DATASET_FILES: Dict[str, str] = {
    'hpc2n':               'hpc2n_5min.csv',
    'google-cluster-v1':   'google-cluster-v1_5min.csv',
    'google-cluster-v2':   'google-cluster-v2_5min.csv',
    'bitbrains':           'bitbrains_5min.csv',
    'azure-public':        'azure-public_5min.csv',
    'alibaba-cluster':     'alibaba-cluster_5min.csv',
    'spitzer':             'spitzer_5min.csv',
    'xmm-newton':          'xmm-newton_5min.csv',
    'parallel-workloads':  'parallel-workloads_5min.csv',
    'caida-2025':          'caida-2025_5min.csv',
}

# Category-level defaults (fallback when CSV unavailable)
_CATEGORY_DEFAULTS: Dict[str, Dict[str, float]] = {
    'hpc':        {'cpu_intensity': 0.75, 'ram_intensity': 0.55, 'task_rate': 0.85, 'burst_prob': 0.05},
    'cloud':      {'cpu_intensity': 0.50, 'ram_intensity': 0.60, 'task_rate': 1.10, 'burst_prob': 0.08},
    'scientific': {'cpu_intensity': 0.60, 'ram_intensity': 0.40, 'task_rate': 0.60, 'burst_prob': 0.18},
    'network':    {'cpu_intensity': 0.35, 'ram_intensity': 0.30, 'task_rate': 1.40, 'burst_prob': 0.12},
}

_DATASET_CATEGORIES: Dict[str, str] = {
    'hpc2n': 'hpc',
    'parallel-workloads': 'hpc',
    'google-cluster-v1': 'cloud',
    'google-cluster-v2': 'cloud',
    'bitbrains': 'cloud',
    'azure-public': 'cloud',
    'alibaba-cluster': 'cloud',
    'spitzer': 'scientific',
    'xmm-newton': 'scientific',
    'caida-2025': 'network',
}


def load_dataset_characteristics(dataset_id: str) -> Dict[str, Any]:
    """
    Load and compute workload characteristics from a real CSV trace.

    Returns a dict with:
      - cpu_intensity    (float 0–1)  : mean CPU utilization across trace
      - ram_intensity    (float 0–1)  : mean RAM utilization across trace
      - task_rate        (float > 0)  : normalized mean task arrival rate (1.0 = baseline)
      - burst_prob       (float 0–1)  : probability of burst event per step
      - cpu_std          (float 0–1)  : CPU volatility (std deviation)
      - ram_std          (float 0–1)  : RAM volatility
      - task_std         (float 0–1)  : task count volatility
      - dataset_id       (str)        : echoed back
      - loaded_from_csv  (bool)       : True if real CSV was used
      - num_rows         (int)        : number of data rows read
    """
    normalized_id = dataset_id.lower().strip()
    csv_file = DATASET_FILES.get(normalized_id)

    if csv_file:
        csv_path = os.path.normpath(os.path.join(_DATA_DIR, csv_file))
        if os.path.isfile(csv_path):
            return _parse_csv(csv_path, normalized_id)

    # Fallback: derive from category defaults
    category = _DATASET_CATEGORIES.get(normalized_id, 'cloud')
    defaults = _CATEGORY_DEFAULTS[category]
    return {
        **defaults,
        'cpu_std': 0.10,
        'ram_std': 0.08,
        'task_std': 0.15,
        'dataset_id': normalized_id,
        'loaded_from_csv': False,
        'num_rows': 0,
    }


def _parse_csv(path: str, dataset_id: str) -> Dict[str, Any]:
    """Read CSV with columns: timestamp, cpu_utilization, ram_utilization, task_count"""
    cpu_vals, ram_vals, task_vals = [], [], []

    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cpu_vals.append(float(row['cpu_utilization']))
                    ram_vals.append(float(row['ram_utilization']))
                    task_vals.append(float(row['task_count']))
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass

    if not cpu_vals:
        category = _DATASET_CATEGORIES.get(dataset_id, 'cloud')
        return {**_CATEGORY_DEFAULTS[category], 'cpu_std': 0.10, 'ram_std': 0.08,
                'task_std': 0.15, 'dataset_id': dataset_id, 'loaded_from_csv': False, 'num_rows': 0}

    cpu_arr  = np.array(cpu_vals,  dtype=np.float32)
    ram_arr  = np.array(ram_vals,  dtype=np.float32)
    task_arr = np.array(task_vals, dtype=np.float32)

    mean_cpu  = float(np.mean(cpu_arr))
    mean_ram  = float(np.mean(ram_arr))
    mean_task = float(np.mean(task_arr))
    std_cpu   = float(np.std(cpu_arr))
    std_ram   = float(np.std(ram_arr))
    std_task  = float(np.std(task_arr))

    # Normalised task rate: ratio vs global baseline of 50 tasks / 5-min interval
    task_rate = float(np.clip(mean_task / 50.0, 0.1, 3.0))

    # Burst probability: fraction of rows where task_count > mean + 1.5*std
    threshold = mean_task + 1.5 * std_task
    burst_prob = float(np.mean(task_arr > threshold))
    burst_prob = float(np.clip(burst_prob, 0.01, 0.40))

    return {
        'cpu_intensity':   float(np.clip(mean_cpu, 0.05, 0.95)),
        'ram_intensity':   float(np.clip(mean_ram, 0.05, 0.95)),
        'task_rate':       task_rate,
        'burst_prob':      burst_prob,
        'cpu_std':         float(np.clip(std_cpu, 0.01, 0.40)),
        'ram_std':         float(np.clip(std_ram, 0.01, 0.40)),
        'task_std':        float(np.clip(std_task / max(mean_task, 1.0), 0.01, 0.50)),
        'dataset_id':      dataset_id,
        'loaded_from_csv': True,
        'num_rows':        len(cpu_vals),
    }
