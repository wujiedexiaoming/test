# -*- coding: utf-8 -*-
"""iTransformer — Inverted Transformer: 对变量维度做 Attention (ICLR 2024)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from config import *

class iTransformer(nn.Module):
    """将时间序列转置: 每个变量成为一个 token, 在变量间做 Attention"""
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8, num_layers=2,
                 dim_feedforward=1024, dropout=0.1, enc_seq_len=24):
        super().__init__(); self.dec_seq_len = dec_seq_len; self.d_model = d_model
        self.input_dim = input_dim; self.enc_seq_len = enc_seq_len
        # 将每个时间步的 d_model 缩到 1 维
        self.time_proj = nn.Linear(enc_seq_len, 1)
        self.var_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.var_proj = nn.Linear(d_model, enc_seq_len)
        self.output_layer = nn.Linear(enc_seq_len, dec_seq_len)

    def forward(self, src, tgt=None, is_training=True):
        # src [B, L, C] → 转置 [B, C, L] → proj [B, C, 1] → embed [B, C, d_model]
        x = src.transpose(1, 2)  # [B, C, L]
        x = self.time_proj(x)    # [B, C, 1]
        x = self.var_embed(x)    # [B, C, d_model]
        # 变量间 Attention
        enc_out = self.encoder(x)  # [B, C, d_model]
        # 投影回时间 → 预测
        time_feat = self.var_proj(enc_out)  # [B, C, L]
        time_feat = time_feat.mean(dim=1)    # [B, L] 聚合变量
        out = self.output_layer(time_feat)   # [B, dec_len]
        return out                           # [B, dec_len]

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
    model = iTransformer(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_layers=2,
                         dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, enc_seq_len=ENC_SEQ_LEN).to(device)
    print(f"\n[iTransformer] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'iTransformer')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[iTransformer] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/iTransformer_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/iTransformer_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
