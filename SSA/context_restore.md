# 对话上下文恢复文件

> 最后更新: 2026-05-15
> 模型: deepseek-v4-pro[1m]
> 项目: 防火涂料实验时序预测 + 滚动预警

---

## 1. 项目背景

- 24个实验: 10原始 + 9新增真实 + 5合成不合格
- 每次 1441 点（5秒间隔，120分钟）
- 17特征: 13炉温 + 2炉压 + 1目标 + 1负载
- **不合格**: value1_avg > 538°C
- 真实不合格: 2023700310 (543.5°C, 119min)
- **数据文件**: `C:\Users\28064\Desktop\2小时室内膨胀型防火涂料试验\姜丝_最终.csv`
- **环境**: `conda activate pytorch` (D:\Anaconda\envs\pytorch, Python 3.9, torch 2.3)

## 2. 关键参数约定

| 变体 | ENC | DEC | FFN | stride | 数据 |
|------|-----|-----|-----|--------|------|
| 短预测系列 (heformer_TSSA*.py) | 24(2min) | 36~48(3~4min) | 1024 | 1 | 姜丝.csv |
| 滚动预警系列 (*_rolling.py) | 120(10min) | 360(30min) | 1024 | TRAIN=2 EVAL=60 | 姜丝_最终.csv |

## 3. 完整文件清单

### 短预测模型 (heformer 系列)
| 文件 | Encoder | 说明 |
|------|---------|------|
| `heformer_TSSA.py` | TSSA | **基准**, stride=1, 原始数据 |
| `heformer_TSSA_TCN.py` | TCN→PE→TSSA (串行) | TCN v1, 效果不好 |
| `heformer_TSSA_TCN_v2.py` | PE→(TCN+Id)→TSSA (并行) | TCN v2, PE之后并行分支 |
| `heformer_TSSA_Freq.py` | PE→(TSSA+Freq)→merge | 频域增强, 并行 |
| `heformer_TSSA_fast.py` | TSSA | 基准+新数据+stride=12(已废弃) |
| `heformer_direct.py` | 标准MHA | 标准Informer, O(n²) |

### 滚动预警模型 (长预测, 120→360)
| 文件 | Encoder | 说明 |
|------|---------|------|
| `heformer_TSSA_rolling_warning.py` | PE→(TCN5层+Id)→TSSA | **当前滚动版**, 含TCN |
| `heformer_TSSA_TCN_rolling.py` | PE→(TCN7层+Id)→TSSA | TCN 7层, 感受野255 |
| `heformer_TSSA_Freq_rolling.py` | PE→(FreqBlock+Id)→TSSA | 频域61bin增强 |
| `heformer_TSSA_early_warning.py` | TSSA→分类头 | 第一版, 已废弃 |

### 其他
| 文件 | 说明 |
|------|------|
| `heformer_direct_TCN.py` | 标准Transformer+TCN |
| `SSA/FDAL_SSA.py` | FDAL-SSA 超参优化 |
| `SSA/value1_avg_analysis.md` | 目标列分析报告 |
| `SSA/rolling_warning_design.md` | 滚动预警设计文档 |
| `SSA/context_restore.md` | **本文件** |

## 4. 滚动预警系统架构

```
每5分钟 (t=10min起):
  [t-120:t] × 17特征 → PE → (增强分支 + Identity) → TSSA×2 → Decoder → 360步预测
  if max(预测) > 538: 报警

Decoder: start_token + 360 zeros → TSSA Self + MHA Cross → 360步输出
```

## 5. 关键技术知识

### stride 原理
- stride=1: 每个时间步取样, 大量冗余但梯度更新充分
- 短窗口(72步)必须stride=1, 长窗口(480步)可用stride≥3
- 训练stride和评估stride可不同

### TSSA 注意力
- O(n)复杂度, 用token W²能量+softmax做重要性分数
- 原版: `nn.Linear(dim,dim)` + L2规范化 + temp学习 + RMS输出
- `attn_drop` 容易定义但未用 (已在TCN_rolling版修复)
- 变量名 `qkv`→`proj` (已在TCN版修复)

### FFN 维度
- **学术标准: 4× d_model**。全项目已统一为1024

### 合成不合格实验
- 2025600991~0995: 基于物理模型(炉温+热响应系数α+时间常数τ)生成
- 5种失效模式: 先天不良/渐进退化/中期退化/涂层开裂/严重失效

### TCN 设计要点
- 放在PE之后(非之前), 并行分支(非串行)
- 层数需匹配输入长度: 120步→7层(感受野255)

### FreqBlock 要点
- FFT在PE之后做(在嵌入空间)
- 幅度增强用 `1.0+tanh` (可增可减), 非 `sigmoid` (只能减)
- 标量gate初始化为0, 训练初期=纯TSSA
- 120步→61个频率bin, 低频刻画升温主趋势

## 6. 已修复的Bug汇总

| 文件 | Bug | 状态 |
|------|-----|------|
| TSSA_TCN.py | TCN层数2→3 | ✅ |
| TSSA_TCN.py | attn_drop未用 | ✅ |
| TSSA_TCN.py | MSE重复计算 | ✅ |
| TSSA_TCN.py | plot_exp_metrics缺失 | ✅ |
| TSSA_TCN.py | PositionalEncoding顺序 | ✅ |
| TSSA_TCN.py | 变量名qkv→proj | ✅ |
| TSSA_Freq.py | sigmoid→tanh | ✅ |
| TSSA_Freq.py | 编辑残留`metrics={` | ✅ |
| TSSA_Freq.py | gate.item()报错 | ✅ |
| rolling_warning.py | 测试集打印尺度修正 | ✅ |
| rolling_warning.py | 自写QKV TSSA→原版 | ✅ |
| rolling_warning.py | FFN 256→1024 | ✅ |

## 7. 运行命令

```bash
source /d/anaconda3/etc/profile.d/conda.sh && conda activate pytorch

# 短预测
python heformer_TSSA.py
python heformer_TSSA_TCN_v2.py
python heformer_TSSA_Freq.py

# 滚动预警 (长预测)
python heformer_TSSA_rolling_warning.py
python heformer_TSSA_TCN_rolling.py
python heformer_TSSA_Freq_rolling.py
```
