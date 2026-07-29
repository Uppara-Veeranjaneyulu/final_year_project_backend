from flask import Blueprint, jsonify, request
import numpy as np

datasets_bp = Blueprint('datasets', __name__)

@datasets_bp.route('/list', methods=['GET'])
def list_datasets():
    return jsonify({
        "datasets": [
            { "id": "google-cluster-v1", "name": "Google Cluster Workload Trace v1", "category": "Cloud", "samples": 670000, "role": "Primary Forecasting & RL Baseline" },
            { "id": "google-cluster-v2", "name": "Google Cluster Trace Version 2.1", "category": "Cloud", "samples": 2400000, "role": "Regime Shift Forecasting Benchmark" },
            { "id": "bitbrains", "name": "Bitbrains GWA-T-12", "category": "Cloud", "samples": 1750000, "role": "Multi-Resource Joint Forecasting" },
            { "id": "azure-public", "name": "Azure Public Dataset 2017", "category": "Cloud", "samples": 2600000, "role": "Cross-Domain Transfer Validation" },
            { "id": "alibaba-cluster", "name": "Alibaba Cluster Trace 2018", "category": "Cloud Microservices", "samples": 1300000, "role": "RL Environment Workload Simulator" },
            { "id": "spitzer", "name": "Spitzer Space Telescope Logs", "category": "Scientific", "samples": 140000, "role": "Irregular Burst Benchmark" },
            { "id": "xmm-newton", "name": "XMM-Newton Observation Logs", "category": "Scientific", "samples": 55000, "role": "Non-Poisson Batch Job Benchmark" },
            { "id": "parallel-workloads", "name": "Parallel Workloads Archive", "category": "HPC", "samples": 500000, "role": "HPC Job Queue Benchmark" },
            { "id": "hpc2n", "name": "HPC2N Workload Dataset", "category": "HPC", "samples": 527371, "role": "Supercomputing Queue Benchmark" },
            { "id": "caida-2025", "name": "CAIDA Internet Traffic 2025", "category": "Network", "samples": 100000000, "role": "External Network Stress Validation" }
        ]
    }), 200

@datasets_bp.route('/<dataset_id>/series', methods=['GET'])
def get_dataset_series(dataset_id):
    length = request.args.get('length', default=100, type=int)
    
    np.random.seed(hash(dataset_id) % 10000)
    t = np.linspace(0, 10, length)
    base = 0.4 + 0.3 * np.sin(t) + 0.1 * np.cos(2.5 * t)
    noise = np.random.normal(0, 0.05, length)
    
    if "spitzer" in dataset_id or "xmm" in dataset_id:
        spikes = (np.random.rand(length) > 0.92) * 0.4
        base += spikes
        
    series = np.clip(base + noise, 0.0, 1.0).tolist()
    return jsonify({
        "dataset_id": dataset_id,
        "length": length,
        "series": series
    }), 200
