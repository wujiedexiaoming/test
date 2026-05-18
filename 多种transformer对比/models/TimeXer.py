# -*- coding: utf-8 -*-
"""TimeXer — 时间交叉 Transformer: 在时间维和变量维交替做 Attention (NeurIPS 2024)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from config import *

class TimeXer(nn.Module):
    """交替 Time-Attention 和 Variable-Attention"""
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=1024, dropout=0.1, enc_seq_len=24):
        super().__init__(); self.dec_seq_len = dec_seq_len
        self.d_model = d_model; self.input_dim = input_dim
        # 时间嵌入: 每时间步 → d_model
        self.time_embed = nn.Linear(input_dim, d_model)
        # 变量嵌入: 每变量 → d_model
        self.var_embed = nn.Linear(enc_seq_len, d_model)
        # 时间维度 Transformer
        t_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                              dropout=dropout, activation='gelu', batch_first=True)
        self.time_enc = nn.TransformerEncoder(t_layer, num_layers=1)
        # 变量维度 Transformer
        v_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                              dropout=dropout, activation='gelu', batch_first=True)
        self.var_enc = nn.TransformerEncoder(v_layer, num_layers=1)
        self.output_layer = nn.Linear(d_model, dec_seq_len)

    def forward(self, src, tgt=None, is_training=True):
        B, L, C = src.shape
        # 时间维度: [B, L, d_model]
        t_emb = self.time_embed(src)
        t_out = self.time_enc(t_emb)  # [B, L, d_model]
        # 转置到变量维度: [B, C, d_model]
        v_emb = self.var_embed(src.transpose(1, 2))  # [B, C, d_model]
        v_out = self.var_enc(v_emb)   # [B, C, d_model]
        # 融合时间+变量特征 → 预测
        fused = t_out.mean(dim=1) + v_out.mean(dim=1)  # [B, d_model]
        out = self.output_layer(fused)  # [B, dec_len]
        return out

def train_model(model, train_loader, val_loader, device, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE_LR, factor=FACTOR)
    best_val_loss, epochs_no_improve = float('inf'), 0
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0
        for src, tgt, target in train_loader:
            src, target = src.to(device), target.to(device)
            optimizer.zero_grad(); loss = criterion(model(src), target.squeeze(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, target = src.to(device), target.to(device)
                val_loss += criterion(model(src), target.squeeze(-1)).item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), f'{RESULTS_DIR}/predictions/{model_name}_best.pth')
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE_ES: break
        if (epoch + 1) % 20 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.6f}")
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/predictions/{model_name}_best.pth'))
    return model

def predict_full_series(model, test_loader, scaler_dict, device):
    target_scaler = scaler_dict['target']
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for src, tgt, target, _ in test_loader:
            src = src.to(device)
            out = model(src); all_preds.append(out.cpu().numpy()); all_targets.append(target.numpy())
    pred = target_scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    true = target_scaler.inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()
    return {'pred': pred, 'true': true, 'rmse': np.sqrt(mean_squared_error(true, pred)),
            'mae': mean_absolute_error(true, pred), 'r2': r2_score(true, pred)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(RANDOM_STATE)
    train_loader, val_loader, test_loader, scaler_dict, _, _, _, input_dim = load_data()
    model = TimeXer(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_layers=2,
                    dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, enc_seq_len=ENC_SEQ_LEN).to(device)
    print(f"\n[TimeXer] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'TimeXer')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[TimeXer] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/TimeXer_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/TimeXer_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
