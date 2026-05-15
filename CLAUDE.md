# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

防火涂料实验时序预测 + 滚动预警系统。对膨胀型防火涂料实验中的器件温度(`value1_avg`)进行多步预测，并在温度超过538°C时提前报警。

- 24个实验（10原始 + 9新增真实 + 5合成不合格），每个1441点（5秒间隔，120分钟）
- 17个特征：13炉温(value0_0~12) + 2炉压(value2_0~1) + 1器件温度(value1_avg, 目标) + 1负载(value3_0)
- 数据文件：`姜丝_最终.csv`（完整版）、`姜丝.csv`（原始10实验）

## 运行环境

```bash
source /d/anaconda3/etc/profile.d/conda.sh && conda activate pytorch
```

运行任意脚本：
```bash
python heformer_TSSA.py              # 基准 TSSA 模型
python heformer_TSSA_TCN.py          # TSSA + TCN
python heformer_TSSA_TCN_v2.py       # TSSA + TCN 并行分支
python heformer_TSSA_rolling_warning.py  # 滚动预警
python heformer_direct.py            # 标准 Informer
python SSA/FDAL_SSA.py               # FDAL-SSA 优化算法基准测试
```

## 架构总览

所有 heformer 系列模型采用 **Informer-style Encoder-Decoder** 架构，核心差异在于 Encoder 的局部特征提取策略：

### 基础架构（heformer_TSSA.py）

```
Encoder: Linear → PositionalEncoding → TSSA × 2 layers → memory
Decoder: Linear → PositionalEncoding → TSSA Self-Attn + 标准MHA Cross-Attn × 2 layers → FFN → Linear(1)
```

- **TSSA (Token Statistics Self-Attention)**：O(n) 复杂度，用 token 能量做重要性分数，替代标准 O(n²) 自注意力
- **Cross-Attention 始终使用标准 MHA**：保证解码质量，不替换为 TSSA
- **推理方式**：非自回归直接预测（Informer风格），decoder 输入为 [start_token + zeros]

### 模型变体谱系

| 文件 | Encoder 增强 | 说明 |
|------|-------------|------|
| `heformer_TSSA.py` | 无 | 基准模型 |
| `heformer_TSSA_TCN.py` | TCN 前置（dilated causal conv） | TCN 在 Linear 之后、PE 之前 |
| `heformer_TSSA_TCN_v2.py` | TCN 并行分支 | TCN 在 PE 之后作为残差并行分支 |
| `heformer_TSSA_Freq.py` | 频域特征 | 加入频率域特征 |
| `heformer_TSSA_fast.py` | stride=12 窗口加速 | 训练时滑动窗口步长 > 1 |
| `heformer_TSSA_revin.py` | RevIN 归一化 | 可逆实例归一化 |
| `heformer_TSSA_SSA.py` | FDAL-SSA 超参优化 | 用 SSA 搜索 d_model、lr、dropout |
| `heformer_TSSA_rolling_warning.py` | 无（滚动预警版） | ENC=120(10min), DEC=360(30min)，每5分钟预测并判警 |
| `heformer_direct*.py` 系列 | 标准 MHA | 同上述变体但使用标准 MultiheadAttention |

### 每个脚本的独立结构

每个脚本都是自包含的（不跨文件导入），内部结构一致：
1. 配置参数（硬编码在文件顶部）
2. 模型定义（TSSA/TCN/Attention 等模块类）
3. 数据集类 `TimeSeriesDataset`
4. 训练 + 验证 + 测试流程
5. 可视化（6张图：training_history, prediction_examples, full_series_comparison, error_analysis, experiment_metrics）
6. 模型/结果保存到 `OUTPUT_DIR`

### SSA/ 目录

FDAL-SSA（Fitness-Distance Adaptive Levy SSA）优化算法研究：
- `FDAL_SSA.py`：改进版麻雀搜索算法，在 F1~F4 基准函数上与原版 SSA 对比
- `test.py`：SSA/ISSA 各变体的消融实验
- `F1F2F3F4figure.py`：绘制对比图
- 核心结论：危险态（20%更新）用距离自适应有向Levy替代高斯随机散步，安全态（80%更新）保持原版公式不变

### TCN 分支设计（v2 版）

```
Encoder: Linear → PE ─┬─ TCN (dil=1/2/4) ─┐
                      └─ Identity ──────────┴─ + → TSSA layers
```

TCN 放在 PE 之后作为残差增强分支，学不到东西也能走原版路径。

## 关键设计约束

- **不要改动 SSA 的安全态/跟随者/警戒者公式**——这些占 80% 更新，改动会破坏收敛
- **Cross-Attention 不要用 TSSA**——标准 MHA 对解码质量至关重要
- **每个脚本 OUTPUT_DIR 必须唯一**——避免覆盖其他变体的训练结果
- **数据归一化**：所有模型使用 MinMaxScaler(-1, 1)，存储在 `scaler.pkl`
- 训练/验证/测试按 experiment_id 做 60/20/20 分层划分（`StratifiedShuffleSplit` 或手工划分）

## 数据集说明

- `姜丝.csv`：10个原始实验（约14410行）
- `姜丝_最终.csv`：24个实验（含9个新增真实 + 5个合成不合格），约34584行
- `姜丝_with不合格.csv`：中间版本
- `ETTh1.csv` / `ETTm1.csv`：Informer 原始基准数据集（ETT=Electricity Transformer Temperature）
