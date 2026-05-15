# Rolling Early Warning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rolling prediction system that monitors fireproof experiments every 5 minutes, predicts the next 30 minutes of device temperature using TSSA Informer, and raises an alarm when the prediction exceeds 538°C.

**Architecture:** One new file `heformer_TSSA_rolling_warning.py` that reuses TSSA modules from `heformer_TSSA.py`. The model is an Informer-style encoder-decoder with TSSA self-attention (O(n)). Training uses MSE loss with teacher forcing. Evaluation runs rolling predictions and reports alarm metrics.

**Tech Stack:** PyTorch, NumPy, Pandas, scikit-learn, matplotlib

---

## File Structure

| File | Responsibility |
|------|---------------|
| `heformer_TSSA_rolling_warning.py` (NEW) | Everything: dataset, model, training, evaluation, visualization |

The TSSA attention modules (`AttentionTSSA`, `TSSALayer`, `HybridDecoderLayer`, `PositionalEncoding`) are copied from `heformer_TSSA.py` to keep the new file self-contained. Model class is `RollingWarningTransformer` — an Informer variant with configurable encoder/decoder sequence lengths.

---

### Task 1: Dataset — Rolling Window Generator

**Files:** Create `D:\Informer2020-main\transformer\heformer_TSSA_rolling_warning.py`

- [ ] **Step 1: Create the file with imports, config, and RollingDataset class**

```python
# -*- coding: utf-8 -*-
"""
TSSA Transformer 滚动预警系统
每5分钟用过去10分钟数据预测未来30分钟器件温度，预测超538度则报警
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 配置 ====================
DATA_PATH = r'C:\Users\28064\Desktop\2小时室内膨胀型防火涂料试验\姜丝_最终.csv'
TARGET_COLUMN = 'value1_avg'
FAIL_THRESHOLD = 538.0
RANDOM_STATE = 42

ENC_SEQ_LEN = 120   # 10分钟历史
DEC_SEQ_LEN = 360   # 30分钟预测
TRAIN_STRIDE = 12    # 训练窗口步长(数据增强)
EVAL_STRIDE = 60     # 评估窗口步长(每5分钟)

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 3
PATIENCE_ES = 20
FACTOR = 0.5

OUTPUT_DIR = 'heformer_rolling_warning'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RollingDataset(Dataset):
    """滑动窗口: [t:t+ENC] → [t+ENC:t+ENC+DEC] target_col"""
    def __init__(self, data, enc_len, dec_len, target_col_idx, stride=12):
        self.data = data
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.target_col_idx = target_col_idx
        self.stride = stride
        self.total_len = enc_len + dec_len
        self.num_samples = max(0, (len(data) - self.total_len) // stride + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        # Encoder input: [enc_len, all_features]
        src = self.data[start:start + self.enc_len, :]
        # Decoder input: [start_token + zeros]
        start_token = self.data[start + self.enc_len - 1, self.target_col_idx]
        tgt = torch.zeros(self.dec_len + 1, 1)
        tgt[0, 0] = start_token
        # Target: future dec_len steps of target column
        tgt_start = start + self.enc_len
        tgt_end = tgt_start + self.dec_len
        target = self.data[tgt_start:tgt_end, self.target_col_idx:self.target_col_idx + 1]
        return torch.FloatTensor(src), tgt, torch.FloatTensor(target)
```

- [ ] **Step 2: Run syntax check**

```bash
source /d/anaconda3/etc/profile.d/conda.sh && conda activate pytorch && cd "D:/Informer2020-main/transformer" && python -c "
import ast
with open('heformer_TSSA_rolling_warning.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

Expected: `Syntax OK`

---

### Task 2: Data Loading with Experiment-Level Split

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add `load_and_preprocess_data()` function after the dataset class

- [ ] **Step 1: Add the data loading function**

```python
def load_and_preprocess_data(file_path, target_column, enc_len, dec_len, train_stride):
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    exp_ids = sorted(df['experiment_id'].unique())

    # Get numeric columns and target index
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'experiment_id']
    target_idx = numeric_cols.index(target_column)

    # Determine experiment labels (for stratified split reference)
    exp_labels = {}
    for eid in exp_ids:
        v = df[df['experiment_id'] == eid][target_column].values
        exp_labels[eid] = int(v.max() > FAIL_THRESHOLD)

    n_fail = sum(exp_labels.values())
    print(f"总实验: {len(exp_ids)} | 不合格: {n_fail} | 合格: {len(exp_ids) - n_fail}")

    # Stratified experiment-level split (60/20/20)
    all_ids = np.array(exp_ids)
    all_labs = np.array([exp_labels[e] for e in exp_ids])

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    for rem_idx, test_idx in sss1.split(all_ids, all_labs):
        rem_ids, test_ids = all_ids[rem_idx], all_ids[test_idx]

    rem_labs = np.array([exp_labels[e] for e in rem_ids])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    for train_idx, val_idx in sss2.split(rem_ids, rem_labs):
        train_ids = rem_ids[train_idx]; val_ids = rem_ids[val_idx]

    train_ids = list(train_ids); val_ids = list(val_ids); test_ids = list(test_ids)
    print(f"训练: {len(train_ids)} | 验证: {len(val_ids)} | 测试: {len(test_ids)}")
    print(f"训练不合格数: {sum(exp_labels[e] for e in train_ids)}")

    # Fit scaler on training data only
    train_raw = df[df['experiment_id'].isin(train_ids)][numeric_cols]
    scaler = MinMaxScaler().fit(train_raw)

    # Build samples
    def build_samples(exp_ids_list, stride):
        samples = []
        for eid in exp_ids_list:
            data = df[df['experiment_id'] == eid][numeric_cols].values
            data_norm = scaler.transform(data)
            if len(data_norm) >= enc_len + dec_len:
                ds = RollingDataset(data_norm, enc_len, dec_len, target_idx, stride)
                samples.extend([ds[i] for i in range(len(ds))])
        return samples

    train_samples = build_samples(train_ids, train_stride)
    val_samples = build_samples(val_ids, train_stride)
    test_samples = build_samples(test_ids, EVAL_STRIDE)  # test uses 5-min stride

    print(f"样本数 — 训练: {len(train_samples)} | 验证: {len(val_samples)} | 测试: {len(test_samples)}")

    train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_samples, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader, test_loader, scaler, numeric_cols, target_idx,
            train_ids, val_ids, test_ids, exp_labels)
```

- [ ] **Step 2: Run syntax check** (same command as Task 1 Step 2)

---

### Task 3: TSSA Attention Modules

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add attention modules after data loading

- [ ] **Step 1: Add PositionalEncoding, AttentionTSSA, TSSALayer, HybridDecoderLayer**

```python
# ==================== 3. TSSA 注意力模块 ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AttentionTSSA(nn.Module):
    """Token Statistics Self-Attention — O(n) 复杂度"""
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Token energy (L2 norm), with softmax normalization
        q_energy = torch.norm(q, dim=-1, keepdim=True)  # [B, heads, N, 1]
        q_weights = torch.softmax(q_energy / self.scale, dim=2)  # normalize across tokens

        # Weighted K, V statistics (single global stat per head)
        k_stats = (k * q_weights).sum(dim=2, keepdim=True)  # [B, heads, 1, head_dim]
        v_stats = (v * q_weights).sum(dim=2, keepdim=True)  # [B, heads, 1, head_dim]

        # Broadcast back: each token modulated by global stats
        out = q * self.scale * (1.0 + k_stats + v_stats)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class TSSALayer(nn.Module):
    """TSSA + FFN with residual connections"""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.attn = AttentionTSSA(d_model, nhead, dropout, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridDecoderLayer(nn.Module):
    """Decoder: TSSA Self-Attn + Standard MHA Cross-Attn"""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = AttentionTSSA(d_model, nhead, dropout, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt = tgt + self.self_attn(self.norm1(tgt))
        residual = tgt
        tgt_normed = self.norm2(tgt)
        attn_out, _ = self.cross_attn(tgt_normed, memory, memory)
        tgt = residual + attn_out
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt
```

- [ ] **Step 2: Run syntax check**

---

### Task 4: RollingWarningTransformer Model

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add model class after attention modules

- [ ] **Step 1: Add the model**

```python
# ==================== 4. 滚动预警模型 ====================

class RollingWarningTransformer(nn.Module):
    """
    Informer-style TSSA Transformer for rolling prediction.
    Encoder: 120 steps × 17 features → memory
    Decoder: start_token + 360 zeros → 360-step prediction of value1_avg
    """
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dec_seq_len = dec_seq_len
        self.d_model = d_model

        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=2000)

        self.encoder_layers = nn.ModuleList([
            TSSALayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            HybridDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt, is_training=True):
        # Encoder
        src_emb = self.pos_encoding(self.encoder_input_layer(src))
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        memory = src_emb

        # Decoder
        tgt_emb = self.pos_encoding(self.decoder_input_layer(tgt))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory)
        output = self.output_layer(tgt_emb)

        if is_training:
            return output[:, :-1, :]   # [B, dec_len, 1]
        else:
            return output[:, 1:, :]    # skip start_token prediction
```

- [ ] **Step 2: Run syntax check**

---

### Task 5: Training & Evaluation Functions

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add training code

- [ ] **Step 1: Add train_model() and evaluate_model()**

```python
# ==================== 5. 训练 ====================

def train_model(model, train_loader, val_loader, device, epochs,
                lr=1e-3, wd=1e-4, patience_lr=3, patience_es=20, factor=0.5):
    print("\n" + "=" * 60)
    print(f"训练: ENC={ENC_SEQ_LEN}步(10min) → DEC={DEC_SEQ_LEN}步(30min)")
    print("=" * 60)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience_lr, factor=factor)

    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for src, tgt, target in train_loader:
            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(src, tgt, is_training=True)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        history['train_loss'].append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                outputs = model(src, tgt, is_training=False)
                val_loss += criterion(outputs, target).item()
        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        scheduler.step(avg_val)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= patience_es:
                print(f"Early stop at epoch {epoch+1}")
                break
    return history
```

- [ ] **Step 2: Run syntax check**

---

### Task 6: Rolling Alarm Evaluation

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add alarm evaluation logic

- [ ] **Step 1: Add rolling_evaluate() function**

```python
# ==================== 6. 滚动预警评估 ====================

@torch.no_grad()
def rolling_evaluate(model, test_ids, df_original, scaler, numeric_cols, target_idx,
                     enc_len, dec_len, eval_stride, device):
    """
    对每个测试实验做滚动预测, 记录报警时间线.
    返回: per-experiment alarm timeline
    """
    target_scaler = MinMaxScaler()
    # Reconstruct target scaler from the full scaler
    model.eval()

    results = {}
    for eid in test_ids:
        exp_data = df_original[df_original['experiment_id'] == eid]
        raw = exp_data[numeric_cols].values
        norm = scaler.transform(raw)
        actual_target = raw[:, target_idx]
        total_len = len(norm)

        # Will this experiment actually exceed 538?
        actual_fails = actual_target.max() > FAIL_THRESHOLD
        actual_exceed_time = None
        if actual_fails:
            actual_exceed_time = np.argmax(actual_target > FAIL_THRESHOLD) * 5 / 60  # minutes

        alarms = []  # list of (time_min, max_pred_value)

        # Rolling prediction from t=enc_len to t=total_len-dec_len
        for t in range(enc_len, total_len - dec_len, eval_stride):
            src = torch.FloatTensor(norm[t - enc_len:t]).unsqueeze(0).to(device)

            # Decoder input: start_token
            start_val = norm[t - 1, target_idx]
            tgt = torch.zeros(1, dec_len + 1, 1)
            tgt[0, 0, 0] = start_val
            tgt = tgt.to(device)

            pred = model(src, tgt, is_training=False)  # [1, dec_len, 1]
            pred_norm = pred[0, :, 0].cpu().numpy()  # [dec_len]

            # Denormalize
            pred_vals = target_scaler.fit_transform(raw[:, target_idx:target_idx+1])
            pred_min = raw[t - enc_len:t, target_idx].min()
            pred_max = raw[t - enc_len:t, target_idx].max()
            # Use the actual scaler info from the training set
            pred_denorm = pred_norm * (pred_max - pred_min) + pred_min

            time_at_pred = t * 5 / 60  # minutes
            max_pred = pred_denorm.max()

            if max_pred > FAIL_THRESHOLD:
                alarms.append((time_at_pred, max_pred))

        # Summarize
        first_alarm = alarms[0][0] if alarms else None
        lead_time = (actual_exceed_time - first_alarm) if (actual_fails and first_alarm) else None

        results[int(eid)] = {
            'actual_fails': actual_fails,
            'actual_exceed_time': actual_exceed_time,
            'first_alarm_time': first_alarm,
            'lead_time_min': lead_time,
            'total_alarms': len(alarms),
            'alarm_timeline': alarms,
        }

    return results


def print_alarm_report(results):
    """打印报警报告"""
    print("\n" + "=" * 70)
    print("滚动预警评估报告")
    print("=" * 70)

    fail_exps = {k: v for k, v in results.items() if v['actual_fails']}
    normal_exps = {k: v for k, v in results.items() if not v['actual_fails']}

    correct_alarms = sum(1 for v in fail_exps.values() if v['first_alarm_time'] is not None)
    false_alarms = sum(1 for v in normal_exps.values() if v['first_alarm_time'] is not None)
    misses = sum(1 for v in fail_exps.values() if v['first_alarm_time'] is None)

    print(f"\n不合格实验 ({len(fail_exps)}个):")
    for eid, r in fail_exps.items():
        status = f"报警! 提前{r['lead_time_min']:.0f}min" if r['first_alarm_time'] else "漏报!"
        print(f"  {eid}: 实际超温@{r['actual_exceed_time']:.0f}min | {status}")

    print(f"\n合格实验 ({len(normal_exps)}个):")
    for eid, r in normal_exps.items():
        status = f"误报! {r['total_alarms']}次" if r['first_alarm_time'] else "正确静默"
        print(f"  {eid}: {status}")

    print(f"\n汇总:")
    print(f"  检出率(Recall): {correct_alarms}/{len(fail_exps)} = {correct_alarms/len(fail_exps):.1%}" if fail_exps else "  N/A")
    print(f"  误报实验数: {false_alarms}/{len(normal_exps)}")
    if fail_exps:
        leads = [r['lead_time_min'] for r in fail_exps.values() if r['lead_time_min'] is not None]
        if leads:
            print(f"  平均预警提前量: {np.mean(leads):.0f} min (范围 [{min(leads):.0f}, {max(leads):.0f}])")
```

- [ ] **Step 2: Run syntax check**

---

### Task 7: Visualization

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add plotting functions

- [ ] **Step 1: Add plot_rolling_prediction()**

```python
# ==================== 7. 可视化 ====================

@torch.no_grad()
def plot_rolling_prediction(model, eid, df_original, scaler, numeric_cols, target_idx,
                             enc_len, dec_len, eval_stride, device, save_dir):
    """绘制单个实验的滚动预测曲线 + 报警标记"""
    exp_data = df_original[df_original['experiment_id'] == eid]
    raw = exp_data[numeric_cols].values
    norm = scaler.transform(raw)
    actual = raw[:, target_idx]
    total_len = len(norm)
    time_all = np.arange(total_len) * 5 / 60

    fig, ax = plt.subplots(figsize=(16, 6))

    # 实际曲线
    ax.plot(time_all, actual, color='blue', linewidth=2, label='Actual')
    ax.axhline(y=FAIL_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'{FAIL_THRESHOLD}C limit')

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, 10))
    pred_idx = 0

    for t in range(enc_len, total_len - dec_len, eval_stride):
        src = torch.FloatTensor(norm[t - enc_len:t]).unsqueeze(0).to(device)
        start_val = norm[t - 1, target_idx]
        tgt = torch.zeros(1, dec_len + 1, 1)
        tgt[0, 0, 0] = start_val
        tgt = tgt.to(device)

        pred = model(src, tgt, is_training=False)[0, :, 0].cpu().numpy()
        pred_time = np.arange(t, t + dec_len) * 5 / 60

        # 反归一化
        pred_min = raw[t - enc_len:t, target_idx].min()
        pred_max = raw[t - enc_len:t, target_idx].max()
        pred_denorm = pred * (pred_max - pred_min) + pred_min

        c = colors[pred_idx % len(colors)]
        ax.plot(pred_time, pred_denorm, color=c, linewidth=0.5, alpha=0.6)

        if pred_denorm.max() > FAIL_THRESHOLD:
            exceed_t = np.argmax(pred_denorm > FAIL_THRESHOLD)
            ax.scatter(pred_time[exceed_t], pred_denorm[exceed_t],
                      color='red', s=60, zorder=5, marker='x')
        pred_idx += 1

    ax.set_xlabel('Time (min)'); ax.set_ylabel('Temperature (C)')
    ax.set_title(f'Experiment {int(eid)} — Rolling Predictions (every {eval_stride*5/60:.0f}min)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'rolling_pred_{int(eid)}.png')
    plt.savefig(fname, dpi=150); plt.close()
    print(f"Saved: {fname}")
```

- [ ] **Step 2: Run syntax check**

---

### Task 8: Main Function — Wire Everything Together

**Files:** Modify `heformer_TSSA_rolling_warning.py` — add `if __name__ == '__main__':` block

- [ ] **Step 1: Add main block**

```python
# ==================== 8. 主函数 ====================

if __name__ == '__main__':
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Encoder: {ENC_SEQ_LEN}步({ENC_SEQ_LEN*5/60:.0f}min) → Decoder: {DEC_SEQ_LEN}步({DEC_SEQ_LEN*5/60:.0f}min)")

    # Load data
    result = load_and_preprocess_data(DATA_PATH, TARGET_COLUMN, ENC_SEQ_LEN, DEC_SEQ_LEN, TRAIN_STRIDE)
    train_loader, val_loader, test_loader, scaler, numeric_cols, target_idx = result[:6]
    train_ids, val_ids, test_ids, exp_labels = result[6:]

    input_dim = len(numeric_cols)
    print(f"Features: {input_dim}")

    # Create model
    model = RollingWarningTransformer(
        input_dim=input_dim, dec_seq_len=DEC_SEQ_LEN,
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, val_loader, device, EPOCHS,
                          lr=LEARNING_RATE, wd=WEIGHT_DECAY,
                          patience_lr=PATIENCE_LR, patience_es=PATIENCE_ES, factor=FACTOR)

    # Test set prediction quality
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    criterion = nn.MSELoss()

    test_loss = 0.0; all_preds = []; all_targets = []
    with torch.no_grad():
        for src, tgt, target in test_loader:
            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            outputs = model(src, tgt, is_training=False)
            test_loss += criterion(outputs, target).item()
            all_preds.append(outputs.cpu().numpy()); all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds).reshape(-1, 1)
    targets = np.concatenate(all_targets).reshape(-1, 1)
    print(f"\n测试集 MSE: {test_loss/len(test_loader):.6f}")
    print(f"测试集 RMSE: {np.sqrt(mean_squared_error(targets, preds)):.4f}")
    print(f"测试集 MAE: {mean_absolute_error(targets, preds)):.4f}")

    # Rolling alarm evaluation
    df_original = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    results = rolling_evaluate(model, test_ids, df_original, scaler, numeric_cols, target_idx,
                               ENC_SEQ_LEN, DEC_SEQ_LEN, EVAL_STRIDE, device)
    print_alarm_report(results)

    # Plot for each test experiment
    for eid in test_ids:
        plot_rolling_prediction(model, eid, df_original, scaler, numeric_cols, target_idx,
                                ENC_SEQ_LEN, DEC_SEQ_LEN, EVAL_STRIDE, device, OUTPUT_DIR)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f)
    with open(os.path.join(OUTPUT_DIR, 'alarm_results.json'), 'w') as f:
        json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'alarm_timeline'}
                   for k, v in results.items()}, f, indent=2)

    print(f"\nDone! Results in {OUTPUT_DIR}/")
```

- [ ] **Step 2: Run the full script**

```bash
source /d/anaconda3/etc/profile.d/conda.sh && conda activate pytorch && cd "D:/Informer2020-main/transformer" && python heformer_TSSA_rolling_warning.py 2>&1
```

Expected: Training runs, shows loss decreasing, prints alarm report.

- [ ] **Step 3: Fix any runtime errors, re-run until clean**

---

### Task 9: Self-Review & Cleanup

- [ ] **Step 1: Verify all tasks produce a working script** — run the final file from scratch
- [ ] **Step 2: Check for unused imports, dead code, commented-out sections**
- [ ] **Step 3: Verify output directory has: best_model.pth, scaler.pkl, history.json, alarm_results.json, rolling_pred_*.png**
