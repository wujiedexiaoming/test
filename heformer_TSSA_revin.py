# -*- coding: utf-8 -*-
"""
终极混合架构：Informer风格 + TSSA（线性注意力） + RevIN（分布归一化）
- Encoder/Decoder-Self: TSSA (O(n)复杂度)
- Decoder-Cross: 标准MultiheadAttention (保证效果)
- RevIN: 处理跨实验分布漂移
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
DATA_PATH = '姜丝.csv'
TARGET_COLUMN = 'value1_avg'
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
RANDOM_STATE = 42

ENC_SEQ_LEN = 24
DEC_SEQ_LEN = 60

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
BATCH_FIRST = True

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 3
PATIENCE_ES = 20
FACTOR = 0.5

OUTPUT_DIR = 'heformer_tssa_revin_final'  # 新目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'history.json')


# ==================== RevIN 模块（新增） ====================

class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    处理跨实验分布漂移的核心模块
    """

    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """
        x: [batch, seq_len, num_features]
        mode: 'norm' 或 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        """计算统计量（沿着时间维度，dim=1）"""
        dim2reduce = tuple(range(1, x.ndim - 1))  # 通常是 (1,)
        self._mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self._std = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        if self.subtract_last:
            self._mean = x[:, -1:, :].detach()

    def _normalize(self, x):
        if self.affine:
            x = (x - self._mean) / (self._std + self.eps)
            x = x * self.affine_weight + self.affine_bias
            return x
        else:
            return (x - self._mean) / (self._std + self.eps)

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x
        else:
            return x * self._std + self._mean


# ==================== TSSA 模块（不变） ====================

class AttentionTSSA(nn.Module):
    """
    Token Statistics Self-Attention (TSSA)
    仅用于Self-Attention，不用于Cross-Attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.dim_head = dim // num_heads

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attend = nn.Softmax(dim=-1)  # 在n维度
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

        # 投影
        return self.to_out(out)


class TSSALayer(nn.Module):
    """TSSA Transformer层（用于Self-Attention）"""

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.tssa = AttentionTSSA(d_model, num_heads=nhead, attn_drop=dropout, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # TSSA + Residual
        x = x + self.tssa(self.norm1(x))
        # FFN + Residual
        x = x + self.ffn(self.norm2(x))
        return x


# ==================== 数据集类 ====================

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, enc_seq_len, dec_seq_len, target_col_idx, exp_id=None):
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_col_idx = target_col_idx
        self.exp_id = exp_id
        self.total_len = enc_seq_len + dec_seq_len

    def __len__(self):
        return len(self.data) - self.total_len + 1

    def __getitem__(self, idx):
        src = self.data[idx:idx + self.enc_seq_len, :]
        start_token = self.data[idx + self.enc_seq_len - 1, self.target_col_idx]
        tgt = torch.zeros(self.dec_seq_len + 1, 1)
        tgt[0, 0] = start_token
        target = self.data[
                     idx + self.enc_seq_len:idx + self.total_len,
                     self.target_col_idx:self.target_col_idx + 1]
        if self.exp_id is not None:
            return torch.FloatTensor(src), tgt, torch.FloatTensor(target), self.exp_id
        return torch.FloatTensor(src), tgt, torch.FloatTensor(target)


def load_and_preprocess_data(file_path, target_column, train_ratio, val_ratio, test_ratio,
                             enc_seq_len, dec_seq_len):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print("=" * 60)
    print(f"【TSSA+RevIN混合架构】输入{enc_seq_len}步 -> 输出{dec_seq_len}步")
    print("=" * 60)

    df = pd.read_csv(file_path)
    exp_ids = df['experiment_id'].unique()
    total_exps = len(exp_ids)
    print(f"总实验数: {total_exps}")

    remaining_ids, test_exp_ids = train_test_split(
        exp_ids, test_size=test_ratio, random_state=RANDOM_STATE, shuffle=True
    )
    val_ratio_relative = val_ratio / (train_ratio + val_ratio)
    train_exp_ids, val_exp_ids = train_test_split(
        remaining_ids, test_size=val_ratio_relative, random_state=RANDOM_STATE, shuffle=True
    )

    print(f"训练: {len(train_exp_ids)} | 验证: {len(val_exp_ids)} | 测试: {len(test_exp_ids)}")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "experiment_id"]
    target_col_idx = numeric_cols.index(target_column)

    train_raw = df[df['experiment_id'].isin(train_exp_ids)][numeric_cols]
    feature_scaler = MinMaxScaler().fit(train_raw)
    target_scaler = MinMaxScaler().fit(train_raw[[target_column]])

    scaler_dict = {
        'feature': feature_scaler,
        'target': target_scaler,
        'target_col_idx': target_col_idx,
        'numeric_cols': numeric_cols
    }
    joblib.dump(scaler_dict, SCALER_SAVE_PATH)

    df_original = df.copy()
    df[numeric_cols] = feature_scaler.transform(df[numeric_cols])
    exp_id_mapping = {int(eid): idx for idx, eid in enumerate(exp_ids)}

    def gen_samples(exp_ids_list, name, include_exp_id=False):
        samples = []
        for eid in exp_ids_list:
            exp_idx = exp_id_mapping[int(eid)]
            data = df[df['experiment_id'] == eid][numeric_cols].values
            if len(data) >= enc_seq_len + dec_seq_len:
                ds = TimeSeriesDataset(data, enc_seq_len, dec_seq_len, target_col_idx,
                                       exp_id=exp_idx if include_exp_id else None)
                samples.extend([ds[i] for i in range(len(ds))])
        print(f"{name}: {len(samples)} samples")
        return samples

    train_samples = gen_samples(train_exp_ids, "Train")
    val_samples = gen_samples(val_exp_ids, "Val")
    test_samples = gen_samples(test_exp_ids, "Test", include_exp_id=True)

    train_loader = DataLoader(train_samples, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_samples, batch_size=BATCH_SIZE, shuffle=False)

    print("数据集划分验证通过！")

    return (train_loader, val_loader, test_loader, scaler_dict, target_col_idx,
            numeric_cols, test_exp_ids, df_original, exp_id_mapping, len(numeric_cols))


# ==================== 混合架构Transformer + RevIN ====================

class HybridTSSATransformerRevIN(nn.Module):
    """
    混合架构 + RevIN：
    - Encoder: TSSA (线性复杂度) + RevIN
    - Decoder Self: TSSA (线性复杂度) + RevIN
    - Decoder Cross: 标准MultiheadAttention (保证效果)
    """

    def __init__(self, input_dim, dec_seq_len, batch_first=True,
                 d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dec_seq_len = dec_seq_len
        self.d_model = d_model
        self.batch_first = batch_first

        # RevIN模块
        self.revin_input = RevIN(input_dim, eps=1e-5, affine=True)
        self.revin_target = RevIN(1, eps=1e-5, affine=True)

        # 输入嵌入
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(1, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout, batch_first=batch_first)

        # Encoder: 纯TSSA
        self.encoder_layers = nn.ModuleList([
            TSSALayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder: TSSA Self-Attn + 标准Cross-Attn
        self.decoder_layers = nn.ModuleList([
            HybridDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt, is_training=True):
        b = src.size(0)

        # RevIN归一化
        src_norm = self.revin_input(src, mode='norm')
        tgt_norm = self.revin_target(tgt, mode='norm')

        # Encoder
        src_emb = self.pos_encoding(self.encoder_input_layer(src_norm))
        for layer in self.encoder_layers:
            src_emb = layer(src_emb)
        memory = src_emb

        # Decoder
        tgt_emb = self.pos_encoding(self.decoder_input_layer(tgt_norm))
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, memory)
        output = self.output_layer(tgt_emb)

        # RevIN反归一化
        output_denorm = self.revin_target(output, mode='denorm')

        if is_training:
            return output_denorm[:, :-1, :]
        else:
            return output_denorm[:, 1:, :]


class HybridDecoderLayer(nn.Module):
    """
    混合Decoder层：
    - Self-Attention: TSSA (线性复杂度)
    - Cross-Attention: 标准MultiheadAttention (保证效果)
    """

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Self-Attention: TSSA
        self.self_attn = AttentionTSSA(d_model, num_heads=nhead, attn_drop=dropout, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-Attention: 标准MultiheadAttention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        # Self-Attention (TSSA)
        tgt = tgt + self.self_attn(self.norm1(tgt))

        # Cross-Attention (标准)
        residual = tgt
        tgt_normed = self.norm2(tgt)
        attn_out, _ = self.cross_attn(tgt_normed, memory, memory)
        tgt = residual + attn_out

        # FFN
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
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


# ==================== 训练和验证 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, scaler_dict):
    print("\n" + "=" * 60)
    print("开始训练 (TSSA+RevIN混合架构)")
    print("=" * 60)

    target_scaler = scaler_dict['target']

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': []
    }

    best_val_loss = float('inf')
    epochs_no_improve = 0

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

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                outputs = model(src, tgt, is_training=False)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # 计算指标
        val_preds_flat = np.concatenate(all_val_preds).reshape(-1, 1)
        val_targets_flat = np.concatenate(all_val_targets).reshape(-1, 1)
        val_pred_orig = target_scaler.inverse_transform(val_preds_flat).flatten()
        val_true_orig = target_scaler.inverse_transform(val_targets_flat).flatten()

        val_rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
        val_mae = mean_absolute_error(val_true_orig, val_pred_orig)
        val_r2 = r2_score(val_true_orig, val_pred_orig)

        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)

        print(f'Epoch [{epoch + 1}/{epochs}] | Train: {avg_train_loss:.4f} | '
              f'Val: {avg_val_loss:.4f} | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE_ES:
                print(f'早停触发！epoch {epoch + 1}')
                break

    return history


def evaluate_model(model, test_loader, scaler_dict, device):
    print("\n测试集评估 (TSSA+RevIN混合架构)")
    target_scaler = scaler_dict['target']
    model.eval()
    all_preds = []
    all_targets = []
    all_exp_ids = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                src, tgt, target, exp_ids = batch
                exp_ids = exp_ids.cpu().numpy()
            else:
                src, tgt, target = batch
                exp_ids = None

            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            outputs = model(src, tgt, is_training=False)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if exp_ids is not None:
                expanded = np.repeat(exp_ids, outputs.size(1), axis=0)
                all_exp_ids.append(expanded)

    preds_flat = np.concatenate(all_preds).reshape(-1, 1)
    targets_flat = np.concatenate(all_targets).reshape(-1, 1)
    pred_original = target_scaler.inverse_transform(preds_flat).flatten()
    true_original = target_scaler.inverse_transform(targets_flat).flatten()

    metrics = {
        'rmse': np.sqrt(mean_squared_error(true_original, pred_original)),
        'mae': mean_absolute_error(true_original, pred_original),
        'r2': r2_score(true_original, pred_original)
    }

    print(f"结果: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # 计算每个实验的指标
    exp_metrics = {}
    if all_exp_ids and len(all_exp_ids) > 0:
        exp_ids_array = np.concatenate(all_exp_ids)
        for exp_idx in np.unique(exp_ids_array):
            mask = (exp_ids_array == exp_idx)
            if not np.any(mask):
                continue
            exp_pred = pred_original[mask]
            exp_true = true_original[mask]
            exp_metrics[str(int(exp_idx))] = {
                'rmse': np.sqrt(mean_squared_error(exp_true, exp_pred)),
                'mae': mean_absolute_error(exp_true, exp_pred),
                'r2': r2_score(exp_true, exp_pred) if len(exp_pred) > 1 else 0.0,
                'sample_count': len(exp_pred)
            }

    return metrics, exp_metrics, pred_original, true_original


# ==================== 可视化函数 ====================

def plot_exp_metrics(exp_metrics, exp_id_mapping):
    """【新增】绘制每个实验的指标对比图"""
    if not exp_metrics:
        print("没有实验级指标可绘制")
        return
    
    # 反转映射：索引 -> 实验ID
    idx_to_exp = {v: k for k, v in exp_id_mapping.items()}
    
    exp_indices = list(exp_metrics.keys())
    exp_rmse = [exp_metrics[e]['rmse'] for e in exp_indices]
    exp_mae = [exp_metrics[e]['mae'] for e in exp_indices]
    exp_r2 = [exp_metrics[e]['r2'] for e in exp_indices]
    exp_labels = [f"实验{idx_to_exp.get(int(e), e)}" for e in exp_indices]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.bar(exp_labels, exp_rmse, color='skyblue')
    plt.title('各实验RMSE (原始尺度)')
    plt.xlabel('实验')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.subplot(1, 3, 2)
    plt.bar(exp_labels, exp_mae, color='lightgreen')
    plt.title('各实验MAE (原始尺度)')
    plt.xlabel('实验')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.subplot(1, 3, 3)
    plt.bar(exp_labels, exp_r2, color='salmon')
    plt.title('各实验R² (原始尺度)')
    plt.xlabel('实验')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'experiment_metrics.png'))
    plt.close()
    print(f"实验级指标图已保存")


def plot_training_history(history):
    """【修改】显示完整的train和val曲线（4个子图）"""
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training and Validation Loss (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 验证集RMSE
    plt.subplot(2, 2, 2)
    plt.plot(history['val_rmse'], label='Val RMSE', color='orange')
    plt.title('Validation RMSE (Original Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # 验证集MAE
    plt.subplot(2, 2, 3)
    plt.plot(history['val_mae'], label='Val MAE', color='orange')
    plt.title('Validation MAE (Original Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    # 验证集R²
    plt.subplot(2, 2, 4)
    plt.plot(history['val_r2'], label='Val R²', color='orange')
    plt.title('Validation R² Score (Original Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    print(f"训练历史图已保存")


def plot_predictions(model, test_loader, scaler_dict, device, num_examples=5):
    """【新增】绘制预测示例图"""
    target_scaler = scaler_dict['target']
    feature_scaler = scaler_dict['feature']
    target_col_idx = scaler_dict['target_col_idx']
    numeric_cols = scaler_dict['numeric_cols']

    model.eval()
    examples_plotted = 0
    plt.figure(figsize=(15, 10))

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if examples_plotted >= num_examples:
                break
            
            if len(batch) == 4:
                src, tgt, target, _ = batch
            else:
                src, tgt, target = batch
            
            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            predictions = model(src, tgt, is_training=False)

            batch_size, seq_len, _ = predictions.shape

            pred_flat = predictions.cpu().numpy().reshape(-1, 1)
            true_flat = target.cpu().numpy().reshape(-1, 1)

            pred_original = target_scaler.inverse_transform(pred_flat).reshape(batch_size, seq_len)
            true_original = target_scaler.inverse_transform(true_flat).reshape(batch_size, seq_len)

            src_flat = src.cpu().numpy().reshape(-1, len(numeric_cols))
            src_original = feature_scaler.inverse_transform(src_flat)[:, target_col_idx].reshape(batch_size, -1)

            plt.subplot(num_examples, 1, examples_plotted + 1)
            plt.plot(range(src.shape[1]), src_original[0], label='History', color='blue', linestyle='--')
            plt.plot(range(src.shape[1], src.shape[1] + seq_len), true_original[0], label='True', color='green')
            plt.plot(range(src.shape[1], src.shape[1] + seq_len), pred_original[0], label='Predicted', color='red',
                     linestyle='--')
            plt.title(f'Example {examples_plotted + 1} (Original Scale)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            examples_plotted += 1

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_examples.png'))
    plt.close()
    print(f"预测示例图已保存")


def plot_error_analysis(pred_original, true_original):
    """【新增】误差分析图"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(true_original, pred_original, alpha=0.5)
    plt.plot([true_original.min(), true_original.max()], [true_original.min(), true_original.max()], 'k--', lw=2)
    plt.title('True vs. Predicted (Original Scale)')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    errors = true_original - pred_original
    plt.hist(errors, bins=50, alpha=0.75)
    plt.title('Error Distribution (Original Scale)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_analysis.png'))
    plt.close()
    print(f"误差分析图已保存")


def plot_full_series_comparison(model, test_exp_ids, df_original, scaler_dict, device,
                                enc_seq_len, dec_seq_len, exp_id_mapping, num_experiments=1):
    """【新增】绘制完整序列对比图"""
    target_scaler = scaler_dict['target']
    feature_scaler = scaler_dict['feature']
    target_col_idx = scaler_dict['target_col_idx']
    numeric_cols = scaler_dict['numeric_cols']

    model.eval()
    selected_exp_ids = test_exp_ids[:num_experiments]
    plt.figure(figsize=(20, 5 * num_experiments))

    for idx, exp_id in enumerate(selected_exp_ids):
        exp_data_original = df_original[df_original['experiment_id'] == exp_id][numeric_cols].values
        seq_len = len(exp_data_original)

        if seq_len < enc_seq_len + dec_seq_len:
            print(f"警告：实验{exp_id}序列长度{seq_len}太短，跳过")
            continue

        exp_data_norm = feature_scaler.transform(exp_data_original)
        exp_idx = exp_id_mapping[int(exp_id)]
        temp_dataset = TimeSeriesDataset(exp_data_norm, enc_seq_len, dec_seq_len, target_col_idx, exp_id=exp_idx)
        temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=False)

        predictions_agg = [[] for _ in range(seq_len)]

        with torch.no_grad():
            for batch_idx, (src, tgt, target, _) in enumerate(temp_loader):
                src = src.to(device)
                tgt = tgt.to(device)
                outputs = model(src, tgt, is_training=False)

                batch_size = outputs.size(0)
                for b in range(batch_size):
                    sample_idx = batch_idx * BATCH_SIZE + b
                    pred_values = outputs[b].cpu().numpy().flatten()
                    for offset, val in enumerate(pred_values):
                        time_idx = sample_idx + enc_seq_len + offset
                        if time_idx < seq_len:
                            predictions_agg[time_idx].append(val)

        pred_series = np.full(seq_len, np.nan)
        for i in range(seq_len):
            if predictions_agg[i]:
                pred_series[i] = np.mean(predictions_agg[i])

        valid_mask = ~np.isnan(pred_series)
        pred_original = np.full(seq_len, np.nan)
        if np.any(valid_mask):
            pred_vals = pred_series[valid_mask].reshape(-1, 1)
            pred_original[valid_mask] = target_scaler.inverse_transform(pred_vals).flatten()

        true_series = exp_data_original[:, target_col_idx]

        plt.subplot(num_experiments, 1, idx + 1)
        plt.plot(range(seq_len), true_series, label='True Values', color='blue',
                 linewidth=2, alpha=0.8, zorder=3)

        if np.any(valid_mask):
            plt.plot(np.arange(seq_len)[valid_mask], pred_original[valid_mask],
                     label='Predicted Values', color='red', linewidth=2,
                     linestyle='--', alpha=0.8, zorder=4)

        plt.axvline(x=enc_seq_len, color='gray', linestyle=':', alpha=0.5,
                    label='Prediction Start', zorder=1)

        plt.title(f'Experiment {exp_id} - Full Series Prediction\n'
                  f'(Blue: True, Red: Predicted, Gray Line: Prediction Start)', fontsize=12)
        plt.xlabel('Time Step')
        plt.ylabel('Value (Original Scale)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        valid_pred = pred_original[valid_mask]
        valid_true = true_series[valid_mask]
        if len(valid_pred) > 0:
            rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
            mae = mean_absolute_error(valid_true, valid_pred)
            r2 = r2_score(valid_true, valid_pred) if len(valid_pred) > 1 else 0.0
            textstr = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_series_comparison.png'), dpi=300)
    plt.close()
    print(f"完整序列对比图已保存 (展示了{len(selected_exp_ids)}个实验)")


# ==================== 主函数 ====================

if __name__ == '__main__':
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)

    temp_df = pd.read_csv(DATA_PATH)
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    if "experiment_id" in numeric_cols:
        numeric_cols.remove("experiment_id")
    input_dim = len(numeric_cols)

    result = load_and_preprocess_data(
        DATA_PATH, TARGET_COLUMN, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        ENC_SEQ_LEN, DEC_SEQ_LEN
    )
    train_loader, val_loader, test_loader = result[0], result[1], result[2]
    scaler_dict = result[3]
    target_col_idx = result[4]
    numeric_cols = result[5]
    test_exp_ids = result[6]
    df_original = result[7]
    exp_id_mapping = result[8]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HybridTSSATransformerRevIN(
        input_dim=input_dim, dec_seq_len=DEC_SEQ_LEN, batch_first=BATCH_FIRST,
        d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    print("【TSSA+RevIN 混合架构说明】")
    print("1. Encoder Self-Attention: TSSA (O(n)线性复杂度)")
    print("2. Decoder Self-Attention: TSSA (O(n)线性复杂度)")
    print("3. Decoder Cross-Attention: 标准MultiheadAttention (保证效果)")
    print("4. RevIN: 处理跨实验分布漂移")
    print("=" * 60)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE_LR, factor=FACTOR)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                          device, EPOCHS, scaler_dict)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    metrics, exp_metrics, pred_original, true_original = evaluate_model(model, test_loader, scaler_dict, device)

    # 保存历史
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f)

    # 绘制所有图表
    plot_training_history(history)
    plot_error_analysis(pred_original, true_original)
    plot_exp_metrics(exp_metrics, exp_id_mapping)
    plot_predictions(model, test_loader, scaler_dict, device, num_examples=5)
    plot_full_series_comparison(model, test_exp_ids, df_original, scaler_dict, device,
                                ENC_SEQ_LEN, DEC_SEQ_LEN, exp_id_mapping, num_experiments=1)

    print(f"\n完成！结果在 {OUTPUT_DIR}")
    print("这是目前最先进的组合：TSSA(效率) + 标准Cross(效果) + RevIN(分布鲁棒)")
