# -*- coding: utf-8 -*-
"""
TSSA Transformer 滚动预警系统
每5分钟用过去10分钟数据预测未来30分钟器件温度，预测超538度则报警

架构: Informer-style Encoder-Decoder
  Encoder: 120步(10min) × 17特征 → TSSA → memory
  Decoder: [start_token + 360 zeros] → TSSA Self + MHA Cross → 360步预测
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
TRAIN_STRIDE = 2    # 训练窗口步长
EVAL_STRIDE = 60     # 评估窗口步长(每5分钟)

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 1024  # 4× d_model，标准Transformer配置
DROPOUT = 0.1

# TCN 并行分支配置 (5层, dil=1/2/4/8/16, 感受野63步≈5.25min)
TCN_NUM_LAYERS = 5
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 3
PATIENCE_ES =100
FACTOR = 0.5

OUTPUT_DIR = 'heformer_rolling_warning'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 2. 数据集 ====================

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
        src = self.data[start:start + self.enc_len, :]
        start_token = self.data[start + self.enc_len - 1, self.target_col_idx]
        tgt = torch.zeros(self.dec_len + 1, 1)
        tgt[0, 0] = start_token
        tgt_end = start + self.enc_len + self.dec_len
        target = self.data[start + self.enc_len:tgt_end,
                           self.target_col_idx:self.target_col_idx + 1]
        return torch.FloatTensor(src), tgt, torch.FloatTensor(target)


def load_and_preprocess_data(file_path, target_column, enc_len, dec_len, train_stride):
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    exp_ids = sorted(df['experiment_id'].unique())

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != 'experiment_id']
    target_idx = numeric_cols.index(target_column)

    exp_labels = {}
    for eid in exp_ids:
        v = df[df['experiment_id'] == eid][target_column].values
        exp_labels[eid] = int(v.max() > FAIL_THRESHOLD)

    n_fail = sum(exp_labels.values())
    print(f"总实验: {len(exp_ids)} | 不合格: {n_fail} | 合格: {len(exp_ids) - n_fail}")

    # 分层实验级划分
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

    # Scaler: 只在训练集上拟合
    train_raw = df[df['experiment_id'].isin(train_ids)][numeric_cols]
    feature_scaler = MinMaxScaler().fit(train_raw)
    target_scaler = MinMaxScaler().fit(train_raw[[target_column]])

    def build_samples(exp_ids_list, stride):
        samples = []
        for eid in exp_ids_list:
            data = df[df['experiment_id'] == eid][numeric_cols].values
            data_norm = feature_scaler.transform(data)
            if len(data_norm) >= enc_len + dec_len:
                ds = RollingDataset(data_norm, enc_len, dec_len, target_idx, stride)
                samples.extend([ds[i] for i in range(len(ds))])
        return samples

    train_samples = build_samples(train_ids, train_stride)
    val_samples = build_samples(val_ids, train_stride)
    test_samples = build_samples(test_ids, EVAL_STRIDE)

    print(f"样本数 — 训练: {len(train_samples)} | 验证: {len(val_samples)} | 测试: {len(test_samples)}")

    train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_samples, batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader, test_loader, feature_scaler, target_scaler,
            numeric_cols, target_idx, train_ids, val_ids, test_ids, exp_labels)


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
    """Token Statistics Self-Attention — 原版实现 (来自 heformer_TSSA.py)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.dim_head = dim // num_heads

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attend = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        b, n, c = x.shape

        # [b, n, dim] -> [b, heads, n, dim_head]
        w = self.qkv(x).view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)

        # 在token维度(n)归一化
        w_normed = torch.nn.functional.normalize(w, p=2, dim=-2, eps=1e-8)
        w_sq = w_normed ** 2

        # 计算Token重要性 Pi [b, heads, n]
        token_energy = torch.sum(w_sq, dim=-1) * self.temp
        Pi = self.attend(token_energy)
        Pi = self.attn_drop(Pi)

        # 归一化Pi
        Pi_normed = Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算加权特征的能量: [b, heads, 1, n] @ [b, heads, n, d] -> [b, heads, 1, d]
        dots = torch.matmul(Pi_normed.unsqueeze(-2), w ** 2)

        # 平方根
        out = torch.sqrt(dots.squeeze(-2) + 1e-8)

        # [b, heads, dim_head] -> [b, dim]
        out = out.view(b, self.heads * self.dim_head)

        # 扩展回原始序列长度 [b, dim] -> [b, n, dim]
        out = out.unsqueeze(1).expand(-1, n, -1)

        return self.to_out(out)


class TSSALayer(nn.Module):
    """TSSA + FFN with pre-norm residuals"""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.attn = AttentionTSSA(d_model, num_heads=nhead, attn_drop=dropout, proj_drop=dropout)
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
        self.self_attn = AttentionTSSA(d_model, num_heads=nhead, attn_drop=dropout, proj_drop=dropout)
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
        attn_out, _ = self.cross_attn(self.norm2(tgt), memory, memory)
        tgt = residual + attn_out
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


# ==================== 3.5. TCN 并行分支 (PE 之后) ====================

class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        x = nn.functional.pad(x, (pad, 0))
        return self.conv(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv = CausalConv1d(channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class ParallelTCNBranch(nn.Module):
    """PE 之后的并行 TCN 增强分支"""
    def __init__(self, d_model, num_layers=5, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(CausalTCNBlock(d_model, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        return x


# ==================== 4. 模型 ====================

class RollingWarningTransformer(nn.Module):
    """Informer-style TSSA Transformer for rolling prediction.
    Encoder: PE ─┬─ TCN (5层, dil=1/2/4/8/16) ─┐
                └─ Identity ───────────────────┴─ + ─→ TSSA layers
    """
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, dropout=0.1,
                 tcn_num_layers=5, tcn_kernel_size=3, tcn_dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dec_seq_len = dec_seq_len
        self.d_model = d_model

        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=2000)

        # TCN 并行分支 (PE 之后)
        self.tcn_branch = ParallelTCNBranch(d_model, tcn_num_layers, tcn_kernel_size, tcn_dropout)

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
        # Encoder: PE → 并行分支(TCN+Identity) → TSSA
        src_emb = self.pos_encoding(self.encoder_input_layer(src))
        tcn_out = self.tcn_branch(src_emb)
        src_emb = src_emb + tcn_out  # 残差融合, TCN学不到就走Identity
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        memory = src_emb

        # Decoder
        tgt_emb = self.pos_encoding(self.decoder_input_layer(tgt))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory)
        output = self.output_layer(tgt_emb)

        if is_training:
            return output[:, :-1, :]
        else:
            return output[:, 1:, :]


# ==================== 5. 训练 ====================

def train_model(model, train_loader, val_loader, target_scaler, device, epochs,
                lr=1e-3, wd=1e-4, patience_lr=3, patience_es=20, factor=0.5):
    print("\n" + "=" * 60)
    print(f"Training: ENC={ENC_SEQ_LEN}步({ENC_SEQ_LEN*5/60:.0f}min) → DEC={DEC_SEQ_LEN}步({DEC_SEQ_LEN*5/60:.0f}min)")
    print("=" * 60)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       patience=patience_lr, factor=factor)
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        # ===== 训练 =====
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

        # ===== 验证 (含原始尺度指标) =====
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                outputs = model(src, tgt, is_training=False)
                val_loss += criterion(outputs, target).item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)

        # 反归一化计算原始尺度指标
        preds_flat = np.concatenate(all_preds).reshape(-1, 1)
        targets_flat = np.concatenate(all_targets).reshape(-1, 1)
        pred_orig = target_scaler.inverse_transform(preds_flat).flatten()
        true_orig = target_scaler.inverse_transform(targets_flat).flatten()

        val_rmse = np.sqrt(mean_squared_error(true_orig, pred_orig))
        val_mae = mean_absolute_error(true_orig, pred_orig)
        val_r2 = r2_score(true_orig, pred_orig)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
              f"RMSE: {val_rmse:.2f}C | MAE: {val_mae:.2f}C | R²: {val_r2:.4f}")

        scheduler.step(avg_val)
        if avg_val < best_loss:
            best_loss = avg_val; no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            no_improve += 1
            if no_improve >= patience_es:
                print(f"早停触发！在 epoch {epoch+1} 停止")
                break
    return history


# ==================== 6. 滚动预警评估 ====================

@torch.no_grad()
def rolling_evaluate(model, test_ids, df_original, feature_scaler, target_scaler,
                     numeric_cols, target_idx, enc_len, dec_len, eval_stride, device):
    """对每个测试实验做滚动预测，记录报警时间线"""
    model.eval()
    results = {}

    for eid in test_ids:
        exp_data = df_original[df_original['experiment_id'] == eid]
        raw = exp_data[numeric_cols].values
        norm = feature_scaler.transform(raw)
        actual_target = raw[:, target_idx]
        total_len = len(norm)

        actual_fails = actual_target.max() > FAIL_THRESHOLD
        actual_exceed_time = None
        if actual_fails:
            actual_exceed_time = np.argmax(actual_target > FAIL_THRESHOLD) * 5 / 60

        alarms = []

        for t in range(enc_len, total_len - dec_len, eval_stride):
            src = torch.FloatTensor(norm[t - enc_len:t]).unsqueeze(0).to(device)
            start_val = norm[t - 1, target_idx]
            tgt = torch.zeros(1, dec_len + 1, 1, device=device)
            tgt[0, 0, 0] = start_val

            pred = model(src, tgt, is_training=False)[0, :, 0].cpu().numpy()

            # 反归一化: 用 target_scaler
            pred_2d = pred.reshape(-1, 1)
            pred_denorm = target_scaler.inverse_transform(pred_2d).flatten()

            time_at_pred = t * 5 / 60
            max_pred = pred_denorm.max()

            if max_pred > FAIL_THRESHOLD:
                alarms.append((time_at_pred, float(max_pred)))

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
    print("\n" + "=" * 70)
    print("滚动预警评估报告")
    print("=" * 70)

    fail_exps = {k: v for k, v in results.items() if v['actual_fails']}
    normal_exps = {k: v for k, v in results.items() if not v['actual_fails']}

    correct_alarms = sum(1 for v in fail_exps.values() if v['first_alarm_time'] is not None)
    false_alarms = sum(1 for v in normal_exps.values() if v['first_alarm_time'] is not None)
    misses = sum(1 for v in fail_exps.values() if v['first_alarm_time'] is None)

    if fail_exps:
        print(f"\n不合格实验 ({len(fail_exps)}个):")
        for eid, r in fail_exps.items():
            if r['first_alarm_time']:
                print(f"  {eid}: 超温@{r['actual_exceed_time']:.0f}min | "
                      f"报警@{r['first_alarm_time']:.0f}min | 提前{r['lead_time_min']:.0f}min")
            else:
                print(f"  {eid}: 超温@{r['actual_exceed_time']:.0f}min | 漏报!")

    if normal_exps:
        print(f"\n合格实验 ({len(normal_exps)}个):")
        for eid, r in normal_exps.items():
            if r['first_alarm_time']:
                print(f"  {eid}: 误报! {r['total_alarms']}次")
            else:
                print(f"  {eid}: 正确静默")

    print(f"\n汇总:")
    if fail_exps:
        recall = correct_alarms / len(fail_exps)
        print(f"  检出率(Recall): {correct_alarms}/{len(fail_exps)} = {recall:.1%}")
    if normal_exps:
        fpr = false_alarms / len(normal_exps)
        print(f"  误报实验: {false_alarms}/{len(normal_exps)} = {fpr:.1%}")
    if fail_exps:
        leads = [r['lead_time_min'] for r in fail_exps.values()
                 if r['lead_time_min'] is not None]
        if leads:
            print(f"  平均提前量: {np.mean(leads):.0f}min [{min(leads):.0f}, {max(leads):.0f}]")


# ==================== 7. 可视化 ====================

@torch.no_grad()
def plot_rolling_prediction(model, eid, df_original, feature_scaler, target_scaler,
                             numeric_cols, target_idx, enc_len, dec_len, eval_stride,
                             device, save_dir):
    """绘制单个实验的滚动预测+报警"""
    exp_data = df_original[df_original['experiment_id'] == eid]
    raw = exp_data[numeric_cols].values
    norm = feature_scaler.transform(raw)
    actual = raw[:, target_idx]
    total_len = len(norm)
    time_all = np.arange(total_len) * 5 / 60

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(time_all, actual, color='blue', linewidth=2, label='Actual')
    ax.axhline(y=FAIL_THRESHOLD, color='red', linestyle='--', linewidth=2,
               label=f'{FAIL_THRESHOLD}C limit')

    # 实际超温标记
    if actual.max() > FAIL_THRESHOLD:
        exceed_t = np.argmax(actual > FAIL_THRESHOLD)
        ax.axvline(x=exceed_t * 5 / 60, color='red', linestyle=':', linewidth=1, alpha=0.5)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, 20))
    pred_idx = 0

    for t in range(enc_len, total_len - dec_len, eval_stride):
        src = torch.FloatTensor(norm[t - enc_len:t]).unsqueeze(0).to(device)
        start_val = norm[t - 1, target_idx]
        tgt = torch.zeros(1, dec_len + 1, 1, device=device)
        tgt[0, 0, 0] = start_val

        pred = model(src, tgt, is_training=False)[0, :, 0].cpu().numpy()
        pred_2d = pred.reshape(-1, 1)
        pred_denorm = target_scaler.inverse_transform(pred_2d).flatten()
        pred_time = np.arange(t, t + dec_len) * 5 / 60

        c = colors[pred_idx % len(colors)]
        ax.plot(pred_time, pred_denorm, color=c, linewidth=0.4, alpha=0.5)

        if pred_denorm.max() > FAIL_THRESHOLD:
            alert_t = np.argmax(pred_denorm > FAIL_THRESHOLD)
            ax.scatter(pred_time[alert_t], pred_denorm[alert_t],
                      color='red', s=50, zorder=5, marker='x')
        pred_idx += 1

    # 标记预测开始线
    ax.axvline(x=enc_len * 5 / 60, color='gray', linestyle=':', alpha=0.4, label='First prediction')

    # 模型预测失败则不在标题标注
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Temperature (C)')
    ax.set_title(f'Experiment {int(eid)} — Rolling Predictions (every 5min, 30min horizon)')
    ax.legend(fontsize=7, loc='upper left'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_dir, f'rolling_pred_{int(eid)}.png')
    plt.savefig(fname, dpi=150); plt.close()
    return fname


# ==================== 8. 主函数 ====================

if __name__ == '__main__':
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"ENC={ENC_SEQ_LEN}步({ENC_SEQ_LEN*5/60:.0f}min) → DEC={DEC_SEQ_LEN}步({DEC_SEQ_LEN*5/60:.0f}min)")
    print(f"预测间隔: {EVAL_STRIDE}步({EVAL_STRIDE*5/60:.0f}min)")

    # 加载数据
    result = load_and_preprocess_data(DATA_PATH, TARGET_COLUMN, ENC_SEQ_LEN, DEC_SEQ_LEN, TRAIN_STRIDE)
    (train_loader, val_loader, test_loader, feature_scaler, target_scaler,
     numeric_cols, target_idx, train_ids, val_ids, test_ids, exp_labels) = result

    input_dim = len(numeric_cols)
    print(f"Features: {input_dim}")

    # 创建模型
    model = RollingWarningTransformer(
        input_dim=input_dim, dec_seq_len=DEC_SEQ_LEN,
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        tcn_num_layers=TCN_NUM_LAYERS, tcn_kernel_size=TCN_KERNEL_SIZE, tcn_dropout=TCN_DROPOUT
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"TCN: {TCN_NUM_LAYERS} layers, dil={[2**i for i in range(TCN_NUM_LAYERS)]}, "
          f"receptive_field={sum(2**i for i in range(TCN_NUM_LAYERS))*(TCN_KERNEL_SIZE-1)+1} steps")

    # 训练
    history = train_model(model, train_loader, val_loader, target_scaler, device, EPOCHS,
                          lr=LEARNING_RATE, wd=WEIGHT_DECAY,
                          patience_lr=PATIENCE_LR, patience_es=PATIENCE_ES, factor=FACTOR)

    # 测试集预测质量
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

    preds_norm = np.concatenate(all_preds).reshape(-1, 1)
    targets_norm = np.concatenate(all_targets).reshape(-1, 1)

    preds_orig = target_scaler.inverse_transform(preds_norm).flatten()
    targets_orig = target_scaler.inverse_transform(targets_norm).flatten()

    test_rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    test_mae = mean_absolute_error(targets_orig, preds_orig)
    test_r2 = r2_score(targets_orig, preds_orig)

    print(f"\nTest set (原始尺度):")
    print(f"  RMSE: {test_rmse:.2f}C | MAE: {test_mae:.2f}C | R²: {test_r2:.4f}")
    print(f"  (归一化参考) MSE: {test_loss/len(test_loader):.6f} | "
          f"RMSE: {np.sqrt(mean_squared_error(targets_norm, preds_norm)):.4f}")

    # 滚动预警评估
    df_original = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    results = rolling_evaluate(model, test_ids, df_original, feature_scaler, target_scaler,
                               numeric_cols, target_idx, ENC_SEQ_LEN, DEC_SEQ_LEN, EVAL_STRIDE, device)
    print_alarm_report(results)

    # 画图
    plot_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    for eid in test_ids:
        fname = plot_rolling_prediction(model, eid, df_original, feature_scaler, target_scaler,
                                        numeric_cols, target_idx, ENC_SEQ_LEN, DEC_SEQ_LEN,
                                        EVAL_STRIDE, device, plot_dir)
        print(f"Saved: {fname}")

    # 保存结果
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history, f)
    serializable = {}
    for k, v in results.items():
        entry = {}
        for kk, vv in v.items():
            if kk == 'alarm_timeline':
                continue
            if isinstance(vv, (np.floating, np.integer, np.bool_)):
                entry[kk] = float(vv) if not isinstance(vv, np.bool_) else bool(vv)
            elif vv is None:
                entry[kk] = None
            else:
                entry[kk] = float(vv) if isinstance(vv, (int, float)) else vv
        serializable[str(k)] = entry
    with open(os.path.join(OUTPUT_DIR, 'alarm_results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nDone! Results in {OUTPUT_DIR}/")
