# -*- coding: utf-8 -*-
"""PatchTST — 补丁时序嵌入 + Transformer Encoder 直接多步预测 (PatchTST, ICLR 2023)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_loader import load_data
from config import *

class PatchTST(nn.Module):
    """将输入序列切为不重叠的 patch，每个 patch 作为 token 输入 Transformer Encoder"""
    def __init__(self, input_dim, dec_seq_len, d_model=256, nhead=8, num_encoder_layers=2,
                 dim_feedforward=1024, dropout=0.1, patch_len=6, stride=6):
        super().__init__(); self.dec_seq_len = dec_seq_len; self.patch_len = patch_len; self.stride = stride
        self.patch_embed = nn.Linear(input_dim * patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, dec_seq_len)

    def _patchify(self, x):
        B, L, C = x.shape
        patches = [x[:, i:i+self.patch_len, :].reshape(B, -1) for i in range(0, L - self.patch_len + 1, self.stride)]
        return torch.stack(patches, dim=1)  # [B, N_patches, C*patch_len]

    def forward(self, src, tgt=None, is_training=True):
        patches = self._patchify(src)  # [B, N, C*P]
        x = self.patch_embed(patches)  # [B, N, d_model]
        enc_out = self.encoder(x)
        # 聚合所有 patch 输出 → 预测序列
        pooled = enc_out.mean(dim=1)   # [B, d_model]
        return self.output_layer(pooled)  # [B, dec_len]

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
    model = PatchTST(input_dim, DEC_SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=2,
                     dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, patch_len=6, stride=6).to(device)
    print(f"\n[PatchTST] Params: {sum(p.numel() for p in model.parameters()):,}")
    model = train_model(model, train_loader, val_loader, device, 'PatchTST')
    results = predict_full_series(model, test_loader, scaler_dict, device)
    print(f"[PatchTST] Test RMSE={results['rmse']:.4f} MAE={results['mae']:.4f} R²={results['r2']:.4f}")
    np.save(f'{PREDICTIONS_DIR}/PatchTST_pred.npy', results['pred'])
    np.save(f'{PREDICTIONS_DIR}/PatchTST_true.npy', results['true'])
    return results

if __name__ == '__main__':
    main()
