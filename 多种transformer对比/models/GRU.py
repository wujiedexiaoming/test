# -*- coding: utf-8 -*-
"""GRU — 双层 GRU Encoder-Decoder 直接多步预测"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from config import *

class GRUEncoderDecoder(nn.Module):
    def __init__(self, input_dim, dec_seq_len, d_model=256, num_layers=2, dropout=0.1):
        super().__init__(); self.dec_seq_len = dec_seq_len; self.d_model = d_model
        self.encoder = nn.GRU(input_dim, d_model, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.GRU(1, d_model, num_layers, dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt, is_training=True):
        _, h = self.encoder(src)
        dec_out, _ = self.decoder(tgt, h)
        out = self.output_layer(dec_out)
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
    target_scaler = scaler_dict['target']
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for src, tgt, target, _ in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, False); all_preds.append(out.cpu().numpy()); all_targets.append(target.numpy())
    pred = target_scaler.inverse_transform(np.concatenate(all_preds).reshape(-1, 1)).flatten()
    true = target_scaler.inverse_transform(np.concatenate(all_targets).reshape(-1, 1)).flatten()
    return {'pred': pred, 'true': true, 'rmse': np.sqrt(mean_squared_error(true, pred)),
            'mae': mean_absolute_error(true, pred), 'r2': r2_score(true, pred)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(RANDOM_STATE)
    train_loader, val_loader, test_loader, scaler_dict, _, _, _, input_dim = load_data()
    model = GRUEncoderDecoder(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, num_layers=2, dropout=DROPOUT).to(device)
    print(f"\n[GRU] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'GRU')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[GRU] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/GRU_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/GRU_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
