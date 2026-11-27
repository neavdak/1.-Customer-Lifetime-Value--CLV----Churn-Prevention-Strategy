#!/usr/bin/env bash
set -euo pipefail

# Day 2 on College DGX: Hyperparameter sweeps, 60-day sequence scaling & high-resolution GPU SHAP.

source .venv/bin/activate

echo "=== [Day 2] Experiment A: Wider 30-day Hybrid Net (batch=2048, lr=5e-4) ==="
python run_pipeline.py \
  --dataset kkbox \
  --step train \
  --seq-len 30 \
  --epochs 45 \
  --batch-size 2048 \
  --lr 0.0005 \
  --dropout 0.30 \
  --conv-filters 96 \
  --gru-units 96 \
  --attn-heads 4 \
  --no-ablation \
  --exp-name day2_30d_wider

echo "=== [Day 2] Preparing 60-Day Behavioral Sequence Tensors ==="
python run_pipeline.py --dataset kkbox --step prep --seq-len 60

echo "=== [Day 2] Experiment B: 60-Day Hybrid Net + Baseline Ablation ==="
python run_pipeline.py \
  --dataset kkbox \
  --step train \
  --seq-len 60 \
  --epochs 45 \
  --batch-size 1024 \
  --lr 0.0008 \
  --dropout 0.25 \
  --conv-filters 96 \
  --gru-units 96 \
  --attn-heads 4 \
  --exp-name day2_60d_ablation

echo "=== [Day 2] Updating CLV, High-Sample GPU SHAP (5000 samples) & Matrix ==="
python run_pipeline.py --dataset kkbox --step clv
python run_pipeline.py --dataset kkbox --step shap --shap-bg 512 --shap-samples 5000
python run_pipeline.py --dataset kkbox --step matrix

echo "=== [Day 2] Experiment History Summary ==="
cat outputs/kkbox/experiment_history.csv
