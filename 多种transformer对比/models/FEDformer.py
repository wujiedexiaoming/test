# -*- coding: utf-8 -*-
"""FEDformer — 频域增强分解 Transformer (ICML 2022), 简化实现"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from config import *

class FourierAttention(nn.Module):
    """频域注意力: 在频域做注意力运算"""
    def __init__(self, d_model, n_modes=12, dropout=0.1):
        super().__init__(); self.n_modes = n_modes; self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        B, L, C = q.shape
        q_f = torch.fft.rfft(q.float(), dim=1)[:, :self.n_modes, :]  # [B, M, C]
        k_f = torch.fft.rfft(k.float(), dim=1)[:, :self.n_modes, :]
        v_f = torch.fft.rfft(v.float(), dim=1)[:, :self.n_modes, :]
        scale = (C ** -0.5)
        attn = torch.softmax((q_f * k_f.conj()).real * scale, dim=-1)
        out_f = attn * v_f  # element-wise, not matmul
        out_t = torch.fft.irfft(out_f, n=L, dim=1)  # [B, L, C]
        return self.out_linear(out_t.type_as(q))

class FEDformer(nn.Module):
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8, num_encoder_layers=2,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__(); self.dec_seq_len = dec_seq_len
        self.enc_embed = nn.Linear(input_dim, d_model)
        self.dec_embed = nn.Linear(1, d_model)
        self.enc_freq_attn = nn.ModuleList([FourierAttention(d_model, n_modes=12, dropout=dropout) for _ in range(num_encoder_layers)])
        self.enc_ffn = nn.ModuleList([nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)) for _ in range(num_encoder_layers)])
        self.enc_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_encoder_layers * 2)])
        self.output_layer = nn.Linear(d_model, dec_seq_len)

    def forward(self, src, tgt=None, is_training=True):
        x = self.enc_embed(src)  # [B, L, d_model]
        for i in range(len(self.enc_freq_attn)):
            x = x + self.enc_freq_attn[i](x, x, x)
            x = self.enc_norm[i * 2](x)
            x = x + self.enc_ffn[i](x)
            x = self.enc_norm[i * 2 + 1](x)
        # 取最后时间步的输出做预测
        out = self.output_layer(x[:, -1, :])  # [B, dec_len]
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
    model = FEDformer(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=2,
                      dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(device)
    print(f"\n[FEDformer] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'FEDformer')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[FEDformer] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/FEDformer_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/FEDformer_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
