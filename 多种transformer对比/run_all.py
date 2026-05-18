# -*- coding: utf-8 -*-
"""批量运行所有对比模型"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Informer, LSTM, GRU, PatchTST, iTransformer, FEDformer, TimeXer
from config import *

MODELS = [
    ('Informer',    Informer),
    ('LSTM',        LSTM),
    ('GRU',         GRU),
    ('PatchTST',    PatchTST),
    ('iTransformer', iTransformer),
    ('FEDformer',   FEDformer),
    ('TimeXer',     TimeXer),
]

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

all_results = {}
total_start = time.time()

for name, module in MODELS:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        result = module.main()
        all_results[name] = {
            'rmse': float(result['rmse']),
            'mae': float(result['mae']),
            'r2': float(result['r2']),
        }
        print(f"[{name}] Done in {(time.time()-start)/60:.1f}min | "
              f"RMSE={result['rmse']:.4f} MAE={result['mae']:.4f} R²={result['r2']:.4f}")
    except Exception as e:
        print(f"[{name}] FAILED: {e}")
        all_results[name] = {'rmse': None, 'mae': None, 'r2': None, 'error': str(e)}

# 保存汇总指标
with open(f'{RESULTS_DIR}/metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"  全部完成! 总耗时: {(time.time()-total_start)/60:.1f}min")
print(f"{'='*60}")

print(f"\n{'Model':<16} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 48)
for name, m in all_results.items():
    if m['rmse'] is not None:
        print(f"{name:<16} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>10.4f}")
    else:
        print(f"{name:<16} {'FAILED':>10}")
