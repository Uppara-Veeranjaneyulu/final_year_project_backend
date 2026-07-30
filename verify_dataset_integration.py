"""
Quick verification script: tests DatasetLoader + CloudSchedulerEnv
with multiple different datasets to confirm each produces distinct results.
Run from the backend directory:
    python verify_dataset_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rl.dataset_loader import load_dataset_characteristics, DATASET_FILES
from app.rl.environment import CloudSchedulerEnv
from app.rl.schedulers import get_scheduler

DATASETS_TO_TEST = [
    'hpc2n',
    'google-cluster-v1',
    'google-cluster-v2',
    'bitbrains',
    'spitzer',
    'xmm-newton',
]

STEPS_PER_TEST = 50

def run_episode(dataset_id, policy='ppo'):
    chars = load_dataset_characteristics(dataset_id)
    env = CloudSchedulerEnv(num_servers=4, max_steps=STEPS_PER_TEST)
    env.load_dataset(chars)
    obs, info = env.reset()

    scheduler = get_scheduler(policy, num_servers=4)
    total_throughput = 0.0
    total_drops = 0
    burst_steps = 0

    for step in range(STEPS_PER_TEST):
        action = scheduler.select_server(obs, num_servers=4)
        obs, _, done, _, metrics = env.step(action)
        total_throughput = metrics['throughput']
        if metrics['dropped']:
            total_drops += 1
        if metrics.get('is_burst', False):
            burst_steps += 1
        if done:
            break

    return {
        'dataset_id': dataset_id,
        'loaded_from_csv': chars['loaded_from_csv'],
        'num_rows': chars.get('num_rows', 0),
        'cpu_intensity': chars['cpu_intensity'],
        'ram_intensity': chars['ram_intensity'],
        'task_rate': chars['task_rate'],
        'burst_prob': chars['burst_prob'],
        'final_throughput': total_throughput,
        'drop_count': total_drops,
        'burst_count': burst_steps,
        'completed_tasks': env.completed_tasks,
    }

print("\n" + "="*80)
print("  DATASET INTEGRATION VERIFICATION")
print("="*80)
print(f"{'Dataset':<25} {'CSV?':^5} {'Rows':^6} {'CPU%':^7} {'RAM%':^7} {'Rate':^6} {'BrstP':^6} {'Thru':^6} {'Drops':^6} {'Bursts':^7}")
print("-"*80)

results = []
for ds_id in DATASETS_TO_TEST:
    r = run_episode(ds_id)
    results.append(r)
    csv_flag = "[OK]" if r['loaded_from_csv'] else "[NO]"
    print(
        f"{r['dataset_id']:<25}"
        f" {csv_flag:^5}"
        f" {r['num_rows']:^6}"
        f" {r['cpu_intensity']*100:^6.1f}%"
        f" {r['ram_intensity']*100:^6.1f}%"
        f" {r['task_rate']:^6.2f}"
        f" {r['burst_prob']*100:^5.1f}%"
        f" {r['final_throughput']:^6.3f}"
        f" {r['drop_count']:^6}"
        f" {r['burst_count']:^7}"
    )

print("="*80)

# Check all throughputs are different (uniqueness test)
throughputs = [round(r['final_throughput'], 3) for r in results]
unique_throughputs = len(set(throughputs))
all_csv_loaded = all(r['loaded_from_csv'] for r in results)

print(f"\n[CHECK] All CSVs loaded from disk : {'YES' if all_csv_loaded else 'PARTIAL -- some used fallback defaults'}")
print(f"[CHECK] Unique throughput values  : {unique_throughputs} / {len(results)} datasets")

if unique_throughputs == len(results):
    print("[PASS] Every dataset produces a genuinely different throughput!")
elif unique_throughputs > len(results) // 2:
    print("[PARTIAL] Most datasets produce different results")
else:
    print("[FAIL] Datasets are producing identical results")

# Check CSV files exist
print(f"\nCSV files in data/processed/:")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
for ds_id, fname in DATASET_FILES.items():
    fpath = os.path.join(data_dir, fname)
    exists = "[OK]" if os.path.isfile(fpath) else "[MISSING]"
    size = f"{os.path.getsize(fpath):,} B" if os.path.isfile(fpath) else ""
    print(f"  {exists}  {fname:<40} {size}")

print()
