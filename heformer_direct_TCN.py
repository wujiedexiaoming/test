# -*- coding: utf-8 -*-
"""
Informer风格的直接多步预测 Transformer + TCN
在 Encoder 侧加入空洞卷积 TCN 模块做局部时序特征提取
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

# ==================== 1. 配置参数 ====================
DATA_PATH = '姜丝.csv'
TARGET_COLUMN = 'value1_avg'
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.6, 0.2, 0.2
RANDOM_STATE = 42

ENC_SEQ_LEN = 24
DEC_SEQ_LEN = 24

D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
BATCH_FIRST = True

# TCN 配置
TCN_NUM_CHANNELS = [256, 256, 256]   # 每层通道数 (与 d_model 对齐)
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 3
PATIENCE_ES = 20
FACTOR = 0.5

OUTPUT_DIR = 'heformer_direct_TCN_prediction'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'history.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 2. 数据集类 ====================

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

        start_token_val = self.data[idx + self.enc_seq_len - 1, self.target_col_idx]
        tgt = torch.zeros(self.dec_seq_len + 1, 1)
        tgt[0, 0] = start_token_val

        target = self.data[
            idx + self.enc_seq_len:idx + self.total_len,
            self.target_col_idx:self.target_col_idx + 1
        ]

        if self.exp_id is not None:
            return torch.FloatTensor(src), tgt, torch.FloatTensor(target), self.exp_id
        return torch.FloatTensor(src), tgt, torch.FloatTensor(target)


def load_and_preprocess_data(file_path, target_column, train_ratio, val_ratio, test_ratio,
                             enc_seq_len, dec_seq_len):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    print("=" * 60)
    print(f"【Informer+TCN风格直接预测】输入{enc_seq_len}步 -> 直接输出{dec_seq_len}步")
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

    print(f"训练: {len(train_exp_ids)}个实验 | 验证: {len(val_exp_ids)}个 | 测试: {len(test_exp_ids)}个")

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


# ==================== 3. PositionalEncoding ====================

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
        else:
            x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


# ==================== 4. TCN 模块 ====================

class TCNBlock(nn.Module):
    """单层空洞卷积 + GELU + Dropout + Residual + LayerNorm"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: [B, C, L]
        residual = self.residual(x)
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # 去掉 causal padding 多出的部分，保持长度不变
        out = self.activation(out)
        out = self.dropout(out)
        out = out + residual
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)  # LayerNorm 在通道维
        return out


class TCNEncoder(nn.Module):
    """堆叠多层 TCNBlock，每层 dilation 翻倍"""
    def __init__(self, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_layers = len(num_channels) - 1 if len(num_channels) > 1 else 1
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = num_channels[i]
            out_ch = num_channels[i + 1] if i + 1 < len(num_channels) else num_channels[-1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, L, C] → [B, C, L] → TCN → [B, L, C]
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)
        return x


# ==================== 5. Informer + TCN 模型 ====================

class InformerTCNTransformer(nn.Module):
    """Informer风格 Transformer，Encoder侧加入TCN做局部特征提取"""
    def __init__(self, input_dim, dec_seq_len, batch_first=True,
                 d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1,
                 tcn_num_channels=None, tcn_kernel_size=3, tcn_dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dec_seq_len = dec_seq_len
        self.d_model = d_model
        self.batch_first = batch_first

        if tcn_num_channels is None:
            tcn_num_channels = [d_model, d_model, d_model]

        # 输入嵌入
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(1, d_model)

        # TCN 编码器（在 PE 之前）
        self.tcn_encoder = TCNEncoder(tcn_num_channels, tcn_kernel_size, tcn_dropout)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, batch_first=batch_first)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            activation='gelu'
        )

        # 输出投影
        self.output_layer = nn.Linear(d_model, 1)

    def make_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, tgt, is_training=True):
        batch_size = src.size(0)

        # Encoder: Linear → TCN → PositionalEncoding → Transformer Encoder
        src_embed = self.encoder_input_layer(src)
        src_embed = self.tcn_encoder(src_embed)
        src_embed = self.positional_encoding(src_embed)

        if not self.batch_first:
            src_embed = src_embed.transpose(0, 1)

        # Decoder: Linear → PositionalEncoding
        tgt_embed = self.decoder_input_layer(tgt)
        tgt_embed = self.positional_encoding(tgt_embed)

        if not self.batch_first:
            tgt_embed = tgt_embed.transpose(0, 1)

        tgt_len = tgt.size(1 if self.batch_first else 0)
        if is_training:
            tgt_mask = self.make_causal_mask(tgt_len).to(src.device)
        else:
            tgt_mask = None

        output = self.transformer(
            src_embed,
            tgt_embed,
            tgt_mask=tgt_mask,
            tgt_is_causal=is_training
        )

        output = self.output_layer(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if is_training:
            return output[:, :-1, :] if self.batch_first else output[:-1, :, :]
        else:
            return output[:, 1:, :] if self.batch_first else output[1:, :, :]


# ==================== 6. 训练和验证 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, scaler_dict):
    print("\n" + "=" * 60)
    print("开始训练 (Informer+TCN 直接多步预测)")
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
        # ===== 训练阶段 =====
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

        # ===== 验证阶段 =====
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

        print(f'Epoch [{epoch+1}/{epochs}] | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'RMSE: {val_rmse:.4f} | '
              f'MAE: {val_mae:.4f} | '
              f'R²: {val_r2:.4f}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE_ES:
                print(f'早停触发！在 epoch {epoch+1} 停止')
                break

    return history


def evaluate_model(model, test_loader, scaler_dict, device):
    print("\n" + "=" * 60)
    print("测试集评估 (Informer+TCN 非自回归预测)")
    print("=" * 60)

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
        'mse': mean_squared_error(true_original, pred_original),
        'rmse': np.sqrt(mean_squared_error(true_original, pred_original)),
        'mae': mean_absolute_error(true_original, pred_original),
        'r2': r2_score(true_original, pred_original)
    }

    print(f"测试集结果:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

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
                'count': len(exp_pred)
            }

    return metrics, exp_metrics, pred_original, true_original


# ==================== 7. 可视化函数 ====================

def plot_exp_metrics(exp_metrics, exp_id_mapping):
    if not exp_metrics:
        print("没有实验级指标可绘制")
        return

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
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange')
    plt.title('Training and Validation Loss (Normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['val_rmse'], label='Val RMSE', color='orange')
    plt.title('Validation RMSE (Original Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['val_mae'], label='Val MAE', color='orange')
    plt.title('Validation MAE (Original Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

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

        plt.title(f'Experiment {exp_id} - Full Series Prediction (Informer+TCN)\n'
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


# ==================== 8. 主函数 ====================

if __name__ == '__main__':
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_STATE)

    temp_df = pd.read_csv(DATA_PATH)
    numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    if "experiment_id" in numeric_cols:
        numeric_cols.remove("experiment_id")
    input_dim = len(numeric_cols)

    print(f"Input dim: {input_dim}, Target: {TARGET_COLUMN}")
    print(f"TCN: channels={TCN_NUM_CHANNELS}, kernel={TCN_KERNEL_SIZE}, "
          f"dilations={[2**i for i in range(len(TCN_NUM_CHANNELS)-1)]}")

    result = load_and_preprocess_data(
        DATA_PATH, TARGET_COLUMN, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        ENC_SEQ_LEN, DEC_SEQ_LEN
    )
    train_loader, val_loader, test_loader = result[0], result[1], result[2]
    scaler_dict, target_col_idx = result[3], result[4]
    numeric_cols, test_exp_ids = result[5], result[6]
    df_original, exp_id_mapping = result[7], result[8]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = InformerTCNTransformer(
        input_dim=input_dim,
        dec_seq_len=DEC_SEQ_LEN,
        batch_first=BATCH_FIRST,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        tcn_num_channels=TCN_NUM_CHANNELS,
        tcn_kernel_size=TCN_KERNEL_SIZE,
        tcn_dropout=TCN_DROPOUT
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE_LR, factor=FACTOR)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, EPOCHS, scaler_dict)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    metrics, exp_metrics, pred_original, true_original = evaluate_model(model, test_loader, scaler_dict, device)

    print("\nFinal Test Results:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")

    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f)

    plot_training_history(history)
    plot_error_analysis(pred_original, true_original)
    plot_exp_metrics(exp_metrics, exp_id_mapping)
    plot_predictions(model, test_loader, scaler_dict, device, num_examples=5)
    plot_full_series_comparison(model, test_exp_ids, df_original, scaler_dict, device,
                                ENC_SEQ_LEN, DEC_SEQ_LEN, exp_id_mapping, num_experiments=1)

    print(f"Done! Results in {OUTPUT_DIR}")
