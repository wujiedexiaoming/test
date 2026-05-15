# 对话上下文恢复文件

> 生成时间: 2026-05-14  
> 模型: deepseek-v4-pro[1m]  
> 项目: 防火涂料实验时序预测 + 滚动预警

---

## 1. 项目背景

- 10次原始防火实验 + 9次新增真实实验 + 5次合成不合格实验 = **24个实验**
- 每次实验 1441 点（5秒间隔，共 120 分钟）
- 17 个特征: 13 个炉温传感器(value0_0~value0_12) + 2 个炉压(value2_0~2_1) + 1 个器件温度(value1_avg, 目标列) + 1 个负载(value3_0)
- **不合格指标**: value1_avg 超过 538°C
- **数据文件**: `C:\Users\28064\Desktop\2小时室内膨胀型防火涂料试验\姜丝_最终.csv`
- **分析报告**: `SSA/value1_avg_analysis.md`

## 2. 已创建的关键文件

### 时序预测模型 (heformer 系列)

| 文件 | 说明 |
|------|------|
| `heformer_TSSA.py` | TSSA Transformer (O(n)注意力)，基准模型 |
| `heformer_TSSA_fast.py` | TSSA + stride=12 窗口加速 + 新数据集 |
| `heformer_TSSA_TCN.py` | TSSA + TCN (膨胀因果卷积) 编码器 |
| `heformer_direct.py` | 标准 Informer Transformer |
| `heformer_direct_TCN.py` | 标准 Transformer + TCN |

### 早期预警系统

| 文件 | 说明 |
|------|------|
| `heformer_TSSA_early_warning.py` | 第一版：30分钟快照 → 二分类(超/不超)。有 bug，已被滚动版取代 |
| `heformer_TSSA_rolling_warning.py` | **滚动预警版**：每5分钟用过去10分钟预测未来30分钟，超538则报警 |

### SSA 目录

| 文件 | 说明 |
|------|------|
| `SSA/test.py` | SSA/ISSA 基准测试 |
| `SSA/FDAL_SSA.py` | FDAL-SSA 改进版 |
| `SSA/FDAL_SSA_原理说明.md` | FDAL-SSA 原理文档 |
| `SSA/value1_avg_analysis.md` | 目标列数据分析报告 |
| `SSA/rolling_warning_design.md` | 滚动预警设计文档 |

### 数据与可视化

| 路径 | 说明 |
|------|------|
| `姜丝_最终.csv` | 24实验完整数据 (含合成不合格) |
| `figure/` | 各实验曲线图 |
| `SSA/figure/` (原) | 原版分析图 |

## 3. 滚动预警系统 (核心成果)

**架构**:
```
每5分钟 (t=10min起):
  过去120步(10min) × 17特征 → TSSA Encoder → TSSA Decoder(start_token+360zeros)
  → 预测360步(30min) value1_avg
  → if max(预测) > 538: 报警
```

**配置**:
- ENC_SEQ_LEN=120, DEC_SEQ_LEN=360
- TRAIN_STRIDE=3 (训练窗口间隔)
- EVAL_STRIDE=60 (评估/预测间隔 = 5分钟)
- D_MODEL=256, DIM_FEEDFORWARD=1024 (4×标准)
- 已换成原版 TSSA (来自 heformer_TSSA.py)

**结果**: 测试集 1 不合格提前 25min 报警, 4 合格零误报

## 4. 关键技术讨论

### stride 原理
- stride=1: 每个时间步取一个窗口，大量冗余但梯度更新多
- stride=12: 每1分钟取一个，去冗余但样本少导致不收敛
- 短窗口 (72步) 需要 stride=1; 长窗口 (480步) stride=12 可行
- 训练 stride 和评估 stride 可不同: 训练密窗口，推理按5min

### TSSA 注意力
- O(n) 复杂度，用 token 能量做重要性分数
- 原版 TSSA: `nn.Linear(dim,dim)` 投影 + W²能量 + softmax + RMS输出
- 滚动版自定义 TSSA 有 softmax 缩放 bug，已修
- `attn_drop` 在原版和 TCN 版都定义但未用，TCN 版已修
- 变量名 `qkv` 在 TCN 版已改为 `proj`

### FFN 维度
- 学术标准: 4× d_model (如 BERT: 768→3072, Informer: 512→2048)
- 滚动版已从 256 改成 1024

### 训练不收敛分析
- 归一化尺度 0.0009 ≠ 不收敛，换算约 22°C RMSE
- 30分钟预测误差 22°C 在物理上合理
- 可做分段评估区分升温段 vs 稳定段误差

## 5. 已修复的 Bug

| 文件 | Bug | 状态 |
|------|-----|------|
| TSSA_TCN.py | TCN 层数只有 2 层 (应为 3) | ✅ 已修 |
| TSSA_TCN.py | attn_drop 定义未使用 | ✅ 已修 |
| TSSA_TCN.py | MSE 重复计算 | ✅ 已修 |
| TSSA_TCN.py | plot_exp_metrics 缺失 | ✅ 已修 |
| TSSA_TCN.py | PositionalEncoding 顺序错 | ✅ 已修 |
| rolling_warning.py | 测试集打印归一化尺度而非°C | ✅ 已修 |
| rolling_warning.py | 自定义 QKV TSSA → 已换回原版 | ✅ 已修 |
| rolling_warning.py | FFN 256→1024 | ✅ 已修 |
| rolling_warning.py | TSSA softmax 缩放反转 | ✅ 已修 |

## 6. 合成不合格实验

5 个合成实验 (2025600991~2025600995)，基于物理模型生成:
- F1 (991): 涂层先天不良, max=555°C, 112min超温
- F2 (992): 涂层渐进退化, max=599°C, 102min超温
- F3 (993): 中期退化, max=594°C, 90min超温
- F4 (994): 涂层开裂, max=631°C, 85min超温
- F5 (995): 严重失效, max=730°C, 58min超温

真实不合格: 2023700310, max=543.5°C, 119min超温 (仅超5.5°C, 持续1.2min)

## 7. 运行命令

```bash
# 激活环境
source /d/anaconda3/etc/profile.d/conda.sh && conda activate pytorch

# 时序预测
python heformer_TSSA.py
python heformer_TSSA_TCN.py

# 滚动预警
python heformer_TSSA_rolling_warning.py
```

## 8. 待做事项

- [ ] 滚动版分段评估 (升温段 vs 稳定段误差)
- [ ] cross-validation 验证滚动预警稳定性
- [ ] 降采样实验 (5s→30s, 减少序列长度)
- [ ] 实验ID嵌入 (让模型区分不同实验特性)
- [ ] TCN 版 LayerNorm → WeightNorm 对比
