import os
import csv
from flask import Blueprint, jsonify, request, abort

datasets_bp = Blueprint('datasets', __name__)

# Absolute path to the processed CSV directory
_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
)

# Static dataset metadata (names, categories, roles are factual)
_DATASET_META = [
    {"id": "google-cluster-v1",  "name": "Google Cluster Workload Trace v1",   "category": "Cloud",              "role": "Primary Forecasting & RL Baseline"},
    {"id": "google-cluster-v2",  "name": "Google Cluster Trace Version 2.1",   "category": "Cloud",              "role": "Regime Shift Forecasting Benchmark"},
    {"id": "bitbrains",          "name": "Bitbrains GWA-T-12",                 "category": "Cloud",              "role": "Multi-Resource Joint Forecasting"},
    {"id": "azure-public",       "name": "Azure Public Dataset 2017",           "category": "Cloud",              "role": "Cross-Domain Transfer Validation"},
    {"id": "alibaba-cluster",    "name": "Alibaba Cluster Trace 2018",          "category": "Cloud Microservices","role": "RL Environment Workload Simulator"},
    {"id": "spitzer",            "name": "Spitzer Space Telescope Logs",        "category": "Scientific",         "role": "Irregular Burst Benchmark"},
    {"id": "xmm-newton",        "name": "XMM-Newton Observation Logs",         "category": "Scientific",         "role": "Non-Poisson Batch Job Benchmark"},
    {"id": "parallel-workloads", "name": "Parallel Workloads Archive",          "category": "HPC",                "role": "HPC Job Queue Benchmark"},
    {"id": "hpc2n",              "name": "HPC2N Workload Dataset",              "category": "HPC",                "role": "Supercomputing Queue Benchmark"},
    {"id": "caida-2025",         "name": "CAIDA Internet Traffic 2025",         "category": "Network",            "role": "External Network Stress Validation"},
]


def _read_csv(dataset_id: str):
    """
    Read processed CSV for dataset_id.
    Returns (cpu_list, ram_list, task_list) or aborts with 404.
    """
    csv_path = os.path.join(_DATA_DIR, f"{dataset_id}_5min.csv")
    if not os.path.isfile(csv_path):
        abort(404, description=f"Dataset '{dataset_id}' not found on disk.")

    cpu_vals, ram_vals, task_vals = [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cpu_vals.append(float(row['cpu_utilization']))
                ram_vals.append(float(row['ram_utilization']))
                task_vals.append(float(row['task_count']))
            except (ValueError, KeyError):
                continue

    if not cpu_vals:
        abort(500, description=f"Dataset '{dataset_id}' CSV is empty or malformed.")

    return cpu_vals, ram_vals, task_vals


@datasets_bp.route('/list', methods=['GET'])
def list_datasets():
    """Return dataset metadata with sample counts and summary stats derived from real CSVs."""
    datasets = []
    for meta in _DATASET_META:
        try:
            cpu_vals, ram_vals, task_vals = _read_csv(meta["id"])
            samples = len(cpu_vals)
            cpu_mean  = round(sum(cpu_vals)  / samples, 4)
            ram_mean  = round(sum(ram_vals)  / samples, 4)
            task_mean = round(sum(task_vals) / samples, 2)
        except Exception:
            # CSV not present — report 0 and skip stats
            samples, cpu_mean, ram_mean, task_mean = 0, None, None, None

        datasets.append({
            **meta,
            "samples":   samples,
            "cpu_mean":  cpu_mean,
            "ram_mean":  ram_mean,
            "task_mean": task_mean,
        })

    return jsonify({"datasets": datasets}), 200


@datasets_bp.route('/<dataset_id>/series', methods=['GET'])
def get_dataset_series(dataset_id):
    """
    Return the real cpu_utilization time series from the processed CSV.
    Optional `length` query param slices the first N rows (default 100).
    """
    length = request.args.get('length', default=100, type=int)

    cpu_vals, _ram_vals, _task_vals = _read_csv(dataset_id)

    # Slice to requested length
    series = cpu_vals[:length]

    return jsonify({
        "dataset_id": dataset_id,
        "length":     len(series),
        "series":     series,
    }), 200
