# Rolling Early Warning System — Design Spec

## Goal
Continuously monitor fireproof coating experiments. Every 5 minutes, use the past 10 minutes of sensor data to predict the next 30 minutes of device temperature. If the predicted temperature exceeds 538°C at any point, raise an alarm.

## Architecture

```
Every 5 min (starting at t=10min):
  Past 120 steps (10min) × 17 features
      ↓
  Informer-style TSSA Encoder
      ↓  memory
  Informer-style TSSA Decoder (start_token + 360 zeros)
      ↓
  Predicted 360 steps (30min) of value1_avg
      ↓
  if max(prediction) > 538 → ALARM
```

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ENC_SEQ_LEN | 120 (10 min) | Enough to capture heating rate + acceleration |
| DEC_SEQ_LEN | 360 (30 min) | Long horizon for early warning |
| PRED_INTERVAL | 60 steps (5 min) | Balance compute vs responsiveness |
| First prediction | t = 120 (10 min) | Need 120 steps of history |

## Model

Based on `heformer_TSSA.py` architecture:
- **Encoder**: Linear → PosEncoding → TSSA×2 → memory
- **Decoder**: Linear → PosEncoding → TSSA-Self + MHA-Cross ×2 → Linear → 1
- **Training**: Teacher forcing with MSE loss
- **Inference**: Direct prediction (Informer style), start_token + zeros

Key difference from original: DEC_SEQ_LEN=360 instead of 24/60, requiring TSSA's O(n) attention.

## Training Data

- Source: `姜丝_最终.csv` (24 experiments)
- Each experiment produces windows: [t:t+120] → [t+120:t+480] value1_avg
- Stride: 60 (matching prediction interval)
- Experiment-level 60/20/20 stratified split

## Evaluation

For each test experiment, run rolling prediction every 5 minutes:

### Metrics per experiment
- **First alarm time**: when predicted curve first exceeds 538
- **Lead time**: (actual exceed time) - (first alarm time), in minutes
- **False alarm**: any alarm on a normal experiment
- **Miss**: failure experiment with no alarm before actual exceed

### Aggregate metrics
- **Recall**: failure experiments correctly alarmed / total failures
- **Precision**: true alarms / total alarms
- **Mean lead time**: average minutes of warning before actual exceed
- **False alarm rate**: normal experiments with any alarm / total normal
- **Per-timestep FPR**: fraction of (normal_experiment × prediction_windows) that trigger false alarm

## Output

- Per-experiment alarm timeline plots (predicted vs actual, with alarm markers)
- Summary table of alarm performance
- Trained model saved to `heformer_rolling_warning/best_model.pth`

## File

- New file: `heformer_TSSA_rolling_warning.py`
- Do NOT modify existing files
