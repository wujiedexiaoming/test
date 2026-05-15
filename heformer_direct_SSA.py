# -*- coding: utf-8 -*-
"""
改进麻雀搜索算法(ISSA) + Transformer 直接多步预测
改进策略：
  1. Tent混沌映射初始化种群（替代随机初始化）
  2. 莱维飞行策略（Lévy Flight）增强发现者全局探索能力
  3. 柯西变异策略（Cauchy Mutation）增强精英个体局部开发能力
"""
import math

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
import gc
import time
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

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_LR = 3
PATIENCE_ES = 20
FACTOR = 0.5

# ==================== ISSA 配置参数 ====================
SSA_N_POP = 10
SSA_MAX_ITER = 15
SSA_EPOCHS_PER_EVAL = 8
SSA_LB = [64, 64, 1e-5]
SSA_UB = [512, 512, 5e-3]

# 莱维飞行参数
LEVY_BETA = 1.5
# 柯西变异参数
CAUCHY_PROB = 0.3  # 每轮触发概率

OUTPUT_DIR = 'heformer_direct_prediction'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
SCALER_SAVE_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, 'history.json')
SSA_RESULT_PATH = os.path.join(OUTPUT_DIR, 'ssa_best_params.json')

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
    print(f"【直接多步预测】输入{enc_seq_len}步 -> 直接输出{dec_seq_len}步")
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


# ==================== 3. Transformer 模型 ====================
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


class InformerStyleTransformer(nn.Module):
    def __init__(self, input_dim, dec_seq_len, batch_first=True,
                 d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dec_seq_len = dec_seq_len
        self.d_model = d_model
        self.batch_first = batch_first
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, batch_first=batch_first)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=batch_first, activation='gelu'
        )
        self.output_layer = nn.Linear(d_model, 1)

    def make_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, src, tgt, is_training=True):
        src_embed = self.encoder_input_layer(src)
        src_embed = self.positional_encoding(src_embed)
        if not self.batch_first:
            src_embed = src_embed.transpose(0, 1)
        tgt_embed = self.decoder_input_layer(tgt)
        tgt_embed = self.positional_encoding(tgt_embed)
        if not self.batch_first:
            tgt_embed = tgt_embed.transpose(0, 1)
        tgt_len = tgt.size(1 if self.batch_first else 0)
        tgt_mask = self.make_causal_mask(tgt_len).to(src.device) if is_training else None
        output = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask, tgt_is_causal=is_training)
        output = self.output_layer(output)
        if not self.batch_first:
            output = output.transpose(0, 1)
        if is_training:
            return output[:, :-1, :] if self.batch_first else output[:-1, :, :]
        else:
            return output[:, 1:, :] if self.batch_first else output[1:, :, :]


# ==================== 4. 改进麻雀搜索算法(ISSA) ====================
class ImprovedSSA:
    """
    改进麻雀搜索算法，包含：
      1. Tent混沌映射初始化
      2. 莱维飞行策略（发现者位置更新）
      3. 柯西变异策略（精英个体后处理）
    """

    def __init__(self, n_pop, max_iter, lb, ub, dim, nhead=8,
                 pd_ratio=0.2, sd_ratio=0.1, st=0.8, seed=42):
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.lb = np.array(lb, dtype=np.float64)
        self.ub = np.array(ub, dtype=np.float64)
        self.dim = dim
        self.nhead = nhead
        self.PD = max(1, int(pd_ratio * n_pop))
        self.SD = max(1, int(sd_ratio * n_pop))
        self.ST = st
        self.seed = seed
        np.random.seed(seed)

        # Tent混沌初始化
        self.X = self._tent_initialize()
        self.fitness = np.full(n_pop, np.inf)
        self.best_fitness = np.inf
        self.best_position = None
        self.history = {'best_fitness': [], 'avg_fitness': [], 'best_params': []}

    # ---------------- Tent混沌映射 ----------------
    def _tent_map(self, x, a=0.5):
        """Tent混沌映射，返回单个值或数组"""
        x = np.asarray(x)
        res = np.where(x < a, x / a, (1 - x) / (1 - a))
        return res

    def _tent_initialize(self):
        """基于Tent混沌映射初始化种群"""
        X = np.zeros((self.n_pop, self.dim))
        for j in range(self.dim):
            # 随机初始种子
            x = np.random.random()
            for i in range(self.n_pop):
                x = self._tent_map(x, a=0.5)
                # 映射到参数边界
                X[i, j] = self.lb[j] + x * (self.ub[j] - self.lb[j])
        # 修复约束
        X = self._repair_position(X)
        print(f"[ISSA] Tent混沌初始化完成，种群范围: [{X.min():.2f}, {X.max():.2f}]")
        return X

    # ---------------- 莱维飞行 ----------------
    def _levy_flight(self, beta=1.5, size=None):
        """
        Mantegna算法生成莱维飞行步长
        size: 如果是整数，返回该长度的数组；如果是元组，返回该shape的数组
        """
        if size is None:
            size = self.dim
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    # ---------------- 位置修复 ----------------
    def _repair_position(self, X):
        """修复越界位置并保证约束"""
        X = np.clip(X, self.lb, self.ub)
        # D_MODEL 必须是 nhead 的倍数
        X[:, 0] = np.round(X[:, 0] / self.nhead) * self.nhead
        X[:, 0] = np.clip(X[:, 0], self.lb[0], self.ub[0])
        # DIM_FEEDFORWARD 取整
        X[:, 1] = np.round(X[:, 1])
        X[:, 1] = np.clip(X[:, 1], self.lb[1], self.ub[1])
        return X

    # ---------------- 适应度评估 ----------------
    def _evaluate_fitness(self, position, input_dim, train_loader, val_loader, device, scaler_dict, epochs):
        d_model = int(position[0])
        dim_feedforward = int(position[1])
        lr = position[2]
        if d_model < self.nhead:
            return 1e6

        print(f"  [ISSA Eval] d_model={d_model}, dim_ff={dim_feedforward}, lr={lr:.6f}")

        model = InformerStyleTransformer(
            input_dim=input_dim, dec_seq_len=DEC_SEQ_LEN, batch_first=BATCH_FIRST,
            d_model=d_model, nhead=self.nhead, num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=dim_feedforward,
            dropout=DROPOUT
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=False)
        target_scaler = scaler_dict['target']
        best_val_rmse = float('inf')

        try:
            for epoch in range(epochs):
                model.train()
                for src, tgt, target in train_loader:
                    src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                    optimizer.zero_grad()
                    outputs = model(src, tgt, is_training=True)
                    loss = criterion(outputs, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                model.eval()
                all_val_preds, all_val_targets = [], []
                val_loss = 0.0
                with torch.no_grad():
                    for src, tgt, target in val_loader:
                        src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                        outputs = model(src, tgt, is_training=False)
                        loss = criterion(outputs, target)
                        val_loss += loss.item()
                        all_val_preds.append(outputs.cpu().numpy())
                        all_val_targets.append(target.cpu().numpy())

                val_preds_flat = np.concatenate(all_val_preds).reshape(-1, 1)
                val_targets_flat = np.concatenate(all_val_targets).reshape(-1, 1)
                val_pred_orig = target_scaler.inverse_transform(val_preds_flat).flatten()
                val_true_orig = target_scaler.inverse_transform(val_targets_flat).flatten()
                val_rmse = np.sqrt(mean_squared_error(val_true_orig, val_pred_orig))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                scheduler.step(val_loss / len(val_loader))
        except Exception as e:
            print(f"  [ISSA Warning] 评估出错: {e}, 返回惩罚值")
            best_val_rmse = 1e6
        finally:
            del model, optimizer, scheduler
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        print(f"  [ISSA Result] Best Val RMSE: {best_val_rmse:.4f}")
        return best_val_rmse

    # ---------------- 主优化循环 ----------------
    def optimize(self, input_dim, train_loader, val_loader, device, scaler_dict, epochs_per_eval):
        print("\n" + "=" * 60)
        print("【改进麻雀搜索算法 ISSA】开始优化超参数")
        print("=" * 60)
        print("改进策略:")
        print("  1. Tent混沌映射初始化种群")
        print("  2. 莱维飞行增强发现者全局探索")
        print("  3. 柯西变异增强精英局部开发")
        print(f"种群数量: {self.n_pop}, 迭代次数: {self.max_iter}")
        print(f"搜索空间: D_MODEL∈[{self.lb[0]},{self.ub[0]}], "
              f"DIM_FF∈[{self.lb[1]},{self.ub[1]}], "
              f"LR∈[{self.lb[2]:.1e},{self.ub[2]:.1e}]")
        print("=" * 60)

        start_time = time.time()

        # 初始评估
        print("\n[ISSA] 初始化种群适应度...")
        for i in range(self.n_pop):
            self.fitness[i] = self._evaluate_fitness(
                self.X[i], input_dim, train_loader, val_loader, device, scaler_dict, epochs_per_eval
            )

        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.X[best_idx].copy()

        # 迭代优化
        for t in range(self.max_iter):
            iter_start = time.time()
            print(f"\n{'=' * 60}")
            print(f"[ISSA] 第 {t + 1}/{self.max_iter} 轮迭代")
            print(f"当前最优: RMSE={self.best_fitness:.4f}, "
                  f"D_MODEL={int(self.best_position[0])}, "
                  f"DIM_FF={int(self.best_position[1])}, "
                  f"LR={self.best_position[2]:.6f}")
            print(f"{'=' * 60}")

            sorted_indices = np.argsort(self.fitness)
            PD_indices = sorted_indices[:self.PD]
            F_indices = sorted_indices[self.PD:]
            X_new = self.X.copy()

            # --- 1. 发现者更新（加入莱维飞行）---
            for i in PD_indices:
                r2 = np.random.random()
                if r2 < self.ST:
                    # 安全：以50%概率使用莱维飞行，50%概率使用原始SSA探索
                    if np.random.random() < 0.5:
                        # 【莱维飞行】向最优方向做重尾步长跳跃
                        levy_step = self._levy_flight(beta=LEVY_BETA, size=self.dim)
                        # 步长按搜索空间尺度自适应缩放，前期较大，后期较小
                        scale = 0.01 * (1 + t / self.max_iter) * (self.ub - self.lb)
                        X_new[i] = self.X[i] + scale * levy_step
                    else:
                        # 原始SSA探索模式
                        alpha = np.random.random()
                        X_new[i] = self.X[i] * np.exp(-alpha / (t + 1))
                else:
                    # 危险：向最优位置快速收敛
                    Q = np.random.normal(0, 1, self.dim)
                    L = np.ones(self.dim)
                    X_new[i] = self.X[i] + Q * L

            # --- 2. 跟随者更新 ---
            for idx, i in enumerate(F_indices):
                if idx < len(F_indices) / 2:
                    A = np.random.choice([-1, 1], self.dim) * np.random.random(self.dim)
                    AA = A / (np.abs(A).sum() + 1e-10)
                    best_pd = self.X[PD_indices[0]]
                    X_new[i] = best_pd + np.abs(self.X[i] - best_pd) * AA
                else:
                    Q = np.random.normal(0, 1, self.dim)
                    X_new[i] = Q * np.exp((self.fitness[i] - self.fitness[sorted_indices[-1]]) /
                                          (np.abs(self.fitness[sorted_indices[-1]]) + 1e-10))

            # --- 3. 警戒者更新 ---
            SD_indices = np.random.choice(self.n_pop, self.SD, replace=False)
            for i in SD_indices:
                f_i = self.fitness[i]
                f_best = self.best_fitness
                f_worst = self.fitness[sorted_indices[-1]]
                if f_i > f_best:
                    beta = np.random.normal(0, 1, self.dim)
                    X_new[i] = self.best_position + beta * np.abs(self.X[i] - self.best_position)
                elif abs(f_i - f_best) < 1e-8:
                    K = np.random.choice([-1, 1]) * np.random.random(self.dim)
                    delta = np.random.random(self.dim)
                    X_new[i] = self.X[i] + K * (delta / (np.abs(self.X[i]) + 1e-10))
                else:
                    X_new[i] = self.best_position + np.random.random(self.dim) * (self.best_position - self.X[i])

            # 修复边界
            X_new = self._repair_position(X_new)

            # --- 4. 评估新位置（贪婪选择） ---
            for i in range(self.n_pop):
                new_fitness = self._evaluate_fitness(
                    X_new[i], input_dim, train_loader, val_loader, device, scaler_dict, epochs_per_eval
                )
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    self.X[i] = X_new[i].copy()
                elif np.random.random() < 0.1:
                    self.fitness[i] = new_fitness
                    self.X[i] = X_new[i].copy()

            # 更新全局最优
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_position = self.X[current_best_idx].copy()

            # --- 5. 【柯西变异】精英个体后处理 ---
            if np.random.random() < CAUCHY_PROB:
                # 柯西变异：对全局最优进行强扰动挖掘
                # 扰动幅度随迭代递减（前期炸得猛，后期炸得轻）
                sigma = 0.1 * (1 - t / self.max_iter) * (self.ub - self.lb)
                cauchy_step = np.random.standard_cauchy(self.dim)
                X_mutant = self.best_position + sigma * cauchy_step
                X_mutant = self._repair_position(X_mutant.reshape(1, -1)).flatten()

                f_mutant = self._evaluate_fitness(
                    X_mutant, input_dim, train_loader, val_loader, device, scaler_dict, epochs_per_eval
                )

                if f_mutant < self.best_fitness:
                    print(f"  [柯西变异] 发现更优解！RMSE: {f_mutant:.4f} < {self.best_fitness:.4f}")
                    self.best_fitness = f_mutant
                    self.best_position = X_mutant.copy()
                    # 将变异个体替换种群中最差个体，维持多样性
                    worst_idx = np.argmax(self.fitness)
                    self.X[worst_idx] = X_mutant.copy()
                    self.fitness[worst_idx] = f_mutant
                else:
                    print(f"  [柯西变异] 未改进，当前最优保持 RMSE: {self.best_fitness:.4f}")

            # 记录历史
            self.history['best_fitness'].append(float(self.best_fitness))
            self.history['avg_fitness'].append(float(np.mean(self.fitness)))
            self.history['best_params'].append({
                'd_model': int(self.best_position[0]),
                'dim_feedforward': int(self.best_position[1]),
                'lr': float(self.best_position[2])
            })

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            remaining = (self.max_iter - t - 1) * (elapsed / (t + 1)) if t > 0 else 0
            print(f"[ISSA] 本轮耗时: {iter_time:.1f}s | 已用: {elapsed / 60:.1f}min | 预计剩余: {remaining / 60:.1f}min")

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"[ISSA] 优化完成! 总耗时: {total_time / 60:.1f}min")
        print(f"[ISSA] 最优超参数:")
        print(f"  D_MODEL: {int(self.best_position[0])}")
        print(f"  DIM_FEEDFORWARD: {int(self.best_position[1])}")
        print(f"  LEARNING_RATE: {self.best_position[2]:.6f}")
        print(f"  最优验证RMSE: {self.best_fitness:.4f}")
        print(f"{'=' * 60}")

        return self.best_position, self.best_fitness, self.history


# ==================== 5. 训练和验证（保持不变）====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, scaler_dict):
    print("\n" + "=" * 60)
    print("开始训练 (直接多步预测)")
    print("=" * 60)
    target_scaler = scaler_dict['target']
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': []}
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

        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_targets = [], []
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

        print(f'Epoch [{epoch + 1}/{epochs}] | '
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
                print(f'早停触发！在 epoch {epoch + 1} 停止')
                break
    return history


def evaluate_model(model, test_loader, scaler_dict, device):
    print("\n" + "=" * 60)
    print("测试集评估 (非自回归预测)")
    print("=" * 60)
    target_scaler = scaler_dict['target']
    model.eval()
    all_preds, all_targets, all_exp_ids = [], [], []
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


# ==================== 6. 可视化函数 ====================
def plot_ssa_history(ssa_history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    iterations = range(1, len(ssa_history['best_fitness']) + 1)

    ax1 = axes[0]
    ax1.plot(iterations, ssa_history['best_fitness'], 'b-o', label='Best RMSE', markersize=5, linewidth=1.5)
    ax1.plot(iterations, ssa_history['avg_fitness'], 'r--s', label='Avg RMSE', markersize=4, alpha=0.6)
    ax1.set_title('ISSA Convergence Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Validation RMSE', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    params = ssa_history['best_params']
    d_models = [p['d_model'] for p in params]
    dim_ffs = [p['dim_feedforward'] for p in params]
    lrs = [p['lr'] for p in params]
    ax2_twin = ax2.twinx()
    ax2.plot(iterations, d_models, 'g-o', label='D_MODEL', markersize=4, linewidth=1.5)
    ax2.plot(iterations, dim_ffs, 'm-s', label='DIM_FF', markersize=4, linewidth=1.5)
    ax2_twin.semilogy(iterations, lrs, 'c-^', label='LR', markersize=4, alpha=0.8)
    ax2.set_title('Hyperparameters Evolution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('D_MODEL / DIM_FF', fontsize=11, color='black')
    ax2_twin.set_ylabel('Learning Rate (log)', fontsize=11, color='c')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] ISSA 优化历史图已保存: {save_path}")


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
        plt.plot(range(seq_len), true_series, label='True Values', color='blue', linewidth=2, alpha=0.8, zorder=3)
        if np.any(valid_mask):
            plt.plot(np.arange(seq_len)[valid_mask], pred_original[valid_mask],
                     label='Predicted Values', color='red', linewidth=2, linestyle='--', alpha=0.8, zorder=4)
        plt.axvline(x=enc_seq_len, color='gray', linestyle=':', alpha=0.5, label='Prediction Start', zorder=1)
        plt.title(f'Experiment {exp_id} - Full Series Prediction', fontsize=12)
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
    print(f"完整序列对比图已保存")


# ==================== 7. 主函数 ====================
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

    # ==================== ISSA 超参数优化 ====================
    print("\n" + "=" * 60)
    print("【阶段1】改进麻雀搜索算法(ISSA)优化超参数")
    print("=" * 60)

    ssa = ImprovedSSA(
        n_pop=SSA_N_POP, max_iter=SSA_MAX_ITER,
        lb=SSA_LB, ub=SSA_UB, dim=3, nhead=NHEAD, seed=RANDOM_STATE
    )

    best_position = None
    best_fitness = None
    ssa_history = None

    try:
        best_position, best_fitness, ssa_history = ssa.optimize(
            input_dim=input_dim, train_loader=train_loader, val_loader=val_loader,
            device=device, scaler_dict=scaler_dict, epochs_per_eval=SSA_EPOCHS_PER_EVAL
        )
    except KeyboardInterrupt:
        print("\n[!] 用户手动中断 ISSA")
        best_position = ssa.best_position
        best_fitness = ssa.best_fitness
        ssa_history = ssa.history
    except Exception as e:
        print(f"\n[!] ISSA 运行出错: {e}")
        best_position = ssa.best_position
        best_fitness = ssa.best_fitness
        ssa_history = ssa.history
    finally:
        if best_position is not None:
            ssa_result = {
                'best_d_model': int(best_position[0]),
                'best_dim_feedforward': int(best_position[1]),
                'best_lr': float(best_position[2]),
                'best_val_rmse': float(best_fitness) if best_fitness is not None else None,
                'ssa_config': {
                    'n_pop': SSA_N_POP, 'max_iter': SSA_MAX_ITER,
                    'epochs_per_eval': SSA_EPOCHS_PER_EVAL, 'lb': SSA_LB, 'ub': SSA_UB
                },
                'history': ssa_history
            }
            with open(SSA_RESULT_PATH, 'w') as f:
                json.dump(ssa_result, f, indent=2, ensure_ascii=False)
            print(f"[✓] ISSA 最优参数已保存: {SSA_RESULT_PATH}")

            if ssa_history and len(ssa_history.get('best_fitness', [])) > 0:
                plot_ssa_history(ssa_history, os.path.join(OUTPUT_DIR, 'ssa_optimization_history.png'))
            else:
                print("[!] ISSA 历史记录为空，无法绘图")

            print(f"\n{'='*60}")
            print("ISSA 阶段结束，最优参数如下（请记录，用于后续训练）：")
            print(f"  D_MODEL: {int(best_position[0])}")
            print(f"  DIM_FEEDFORWARD: {int(best_position[1])}")
            print(f"  LEARNING_RATE: {best_position[2]:.6f}")
            print(f"{'='*60}")
        else:
            print("[✗] ISSA 未能找到任何有效参数")

    # 如果ISSA成功，继续完整训练
    if best_position is not None:
        BEST_D_MODEL = int(best_position[0])
        BEST_DIM_FEEDFORWARD = int(best_position[1])
        BEST_LR = float(best_position[2])

        print("\n" + "=" * 60)
        print("【阶段2】使用ISSA最优参数进行完整训练")
        print(f"最优参数: D_MODEL={BEST_D_MODEL}, DIM_FF={BEST_DIM_FEEDFORWARD}, LR={BEST_LR:.6f}")
        print("=" * 60)

        model = InformerStyleTransformer(
            input_dim=input_dim, dec_seq_len=DEC_SEQ_LEN, batch_first=BATCH_FIRST,
            d_model=BEST_D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=BEST_DIM_FEEDFORWARD,
            dropout=DROPOUT
        ).to(device)

        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=BEST_LR, weight_decay=WEIGHT_DECAY)
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
        plot_predictions(model, test_loader, scaler_dict, device, num_examples=5)
        plot_full_series_comparison(model, test_exp_ids, df_original, scaler_dict, device,
                                    ENC_SEQ_LEN, DEC_SEQ_LEN, exp_id_mapping, num_experiments=1)

        print(f"\nDone! Results in {OUTPUT_DIR}")