# -*- coding: utf-8 -*-
"""Informer — 标准 Transformer 直接多步预测"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data, TimeSeriesDataset
from torch.utils.data import DataLoader
from config import *

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
        pe = pe.unsqueeze(0) if batch_first else pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :] if self.batch_first else x + self.pe[:x.size(0), :, :])

class Informer(nn.Module):
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=1024, dropout=0.1, batch_first=True):
        super().__init__()
        self.dec_seq_len = dec_seq_len; self.batch_first = batch_first
        self.encoder_input = nn.Linear(input_dim, d_model)
        self.decoder_input = nn.Linear(1, d_model)
        self.pe = PositionalEncoding(d_model, dropout, batch_first=batch_first)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=batch_first, activation='gelu')
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt, is_training=True):
        src_emb = self.pe(self.encoder_input(src))
        tgt_emb = self.pe(self.decoder_input(tgt))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(src.device) if is_training else None
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                               tgt_is_causal=is_training if hasattr(torch.nn.Transformer, '__init__') else None)
        out = self.output_layer(out)
        return out[:, :-1, :] if is_training else out[:, 1:, :]

def train_model(model, train_loader, val_loader, device, model_name):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=PATIENCE_LR, factor=FACTOR)
    best_val_loss, epochs_no_improve = float('inf'), 0
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0
        for src, tgt, target in train_loader:
            src, tgt, target = src.to(device), tgt.to(device), target.to(device)
            optimizer.zero_grad(); loss = criterion(model(src, tgt, True), target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, tgt, target = src.to(device), tgt.to(device), target.to(device)
                val_loss += criterion(model(src, tgt, False), target).item()
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
    target_scaler = scaler_dict['target']; feature_scaler = scaler_dict['feature']
    target_col_idx = scaler_dict['target_col_idx']; numeric_cols = scaler_dict['numeric_cols']
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for src, tgt, target, _ in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, False); all_preds.append(out.cpu().numpy()); all_targets.append(target.numpy())
    pred = target_scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    true = target_scaler.inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(true, pred)); mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    return {'pred': pred, 'true': true, 'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(RANDOM_STATE)
    train_loader, val_loader, test_loader, scaler_dict, test_exp_ids, df_original, exp_id_mapping, input_dim = load_data()
    model = Informer(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS,
                     num_decoder_layers=NUM_DECODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(device)
    print(f"\n[Informer] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'Informer')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[Informer] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/Informer_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/Informer_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
