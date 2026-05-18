# -*- coding: utf-8 -*-
"""汇总对比图: 所有模型的预测曲线 vs 真实曲线"""
import numpy as np, matplotlib.pyplot as plt, os, json
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PREDICTIONS_DIR = 'predictions'
MODEL_NAMES = ['Informer', 'LSTM', 'GRU', 'PatchTST', 'iTransformer', 'FEDformer', 'TimeXer']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
LINE_STYLES = ['--', '-.', ':', '--', '-.', ':', '--']

def plot_comparison(num_experiments=2):
    fig, axes = plt.subplots(num_experiments, 1, figsize=(16, 5 * num_experiments))
    if num_experiments == 1:
        axes = [axes]

    for exp_idx in range(num_experiments):
        ax = axes[exp_idx]
        ax.grid(True, alpha=0.3)

        # 每实验 1441 个时间点, 选前 200 个展示
        offset = exp_idx * 1441
        plot_len = 200

        # 加载真实曲线（所有模型用同一份 true）
        for model_name in MODEL_NAMES:
            true_path = f'{PREDICTIONS_DIR}/{model_name}_true.npy'
            if os.path.exists(true_path):
                true = np.load(true_path)
                ax.plot(range(plot_len), true[offset:offset + plot_len], 'k-',
                        linewidth=2, label='True', alpha=0.9, zorder=10)
                break

        # 加载各模型预测
        for idx, model_name in enumerate(MODEL_NAMES):
            pred_path = f'{PREDICTIONS_DIR}/{model_name}_pred.npy'
            if os.path.exists(pred_path):
                pred = np.load(pred_path)
                color = COLORS[idx % len(COLORS)]
                ls = LINE_STYLES[idx % len(LINE_STYLES)]
                ax.plot(range(plot_len), pred[offset:offset + plot_len],
                        color=color, linestyle=ls, linewidth=1.5, label=model_name, alpha=0.8)

        ax.set_title(f'Experiment {exp_idx + 1} — 多模型预测对比', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('value1_avg', fontsize=11)
        ax.legend(loc='upper right', ncol=4, fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{PREDICTIONS_DIR}/../model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 对比图已保存: 多种transformer对比/model_comparison.png")

def print_metrics_table():
    metrics_path = 'metrics.json'
    if not os.path.exists(metrics_path):
        print("[!] metrics.json 不存在, 请先运行 run_all.py")
        return
    with open(metrics_path) as f:
        data = json.load(f)
    print(f"\n{'Model':<16} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 48)
    for name, m in data.items():
        if m.get('rmse') is not None:
            print(f"{name:<16} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>10.4f}")
        else:
            print(f"{name:<16} {'FAILED':>10}")

if __name__ == '__main__':
    plot_comparison(num_experiments=2)
    print_metrics_table()
