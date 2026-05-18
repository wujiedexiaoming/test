# -*- coding: utf-8 -*-
"""多模型对比 — 共享数据加载器"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import *


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


def load_data():
    print("=" * 60)
    print(f"【多模型对比数据加载】输入{ENC_SEQ_LEN}步 -> 输出{DEC_SEQ_LEN}步")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    exp_ids = df['experiment_id'].unique()
    print(f"总实验数: {len(exp_ids)}")

    remaining_ids, test_exp_ids = train_test_split(
        exp_ids, test_size=TEST_RATIO, random_state=RANDOM_STATE, shuffle=True
    )
    val_ratio_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_exp_ids, val_exp_ids = train_test_split(
        remaining_ids, test_size=val_ratio_relative, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"训练: {len(train_exp_ids)} | 验证: {len(val_exp_ids)} | 测试: {len(test_exp_ids)}")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "experiment_id"]
    target_col_idx = numeric_cols.index(TARGET_COLUMN)

    train_raw = df[df['experiment_id'].isin(train_exp_ids)][numeric_cols]
    feature_scaler = MinMaxScaler().fit(train_raw)
    target_scaler = MinMaxScaler().fit(train_raw[[TARGET_COLUMN]])

    scaler_dict = {
        'feature': feature_scaler, 'target': target_scaler,
        'target_col_idx': target_col_idx, 'numeric_cols': numeric_cols
    }

    df_original = df.copy()
    df[numeric_cols] = feature_scaler.transform(df[numeric_cols])
    exp_id_mapping = {int(eid): idx for idx, eid in enumerate(exp_ids)}

    def gen_samples(exp_ids_list, name, include_exp_id=False):
        samples = []
        for eid in exp_ids_list:
            exp_idx = exp_id_mapping[int(eid)]
            data = df[df['experiment_id'] == eid][numeric_cols].values
            if len(data) >= ENC_SEQ_LEN + DEC_SEQ_LEN:
                ds = TimeSeriesDataset(data, ENC_SEQ_LEN, DEC_SEQ_LEN, target_col_idx,
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

    return (train_loader, val_loader, test_loader, scaler_dict,
            test_exp_ids, df_original, exp_id_mapping, len(numeric_cols))
