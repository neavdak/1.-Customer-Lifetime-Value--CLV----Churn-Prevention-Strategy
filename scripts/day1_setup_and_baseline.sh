#!/usr/bin/env bash
set -euo pipefail

# Day 1 on College DGX: Environment setup, Telco smoke test, KKBox download & 30-day baseline.

echo "=== [Day 1] Checking NVIDIA GPU ==="
nvidia-smi || true

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "=== [Day 1] Verifying TensorFlow GPU ==="
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"

echo "=== [Day 1] 1-Minute Telco Pipeline Smoke Test ==="
python run_pipeline.py --dataset telco --step all --epochs 5 --exp-name day1_telco_smoke

if [ ! -f "data/raw/kkbox/train_v2.csv" ] && [ ! -f "data/raw/kkbox/train.csv" ]; then
  echo "=== [Day 1] Downloading KKBox Dataset ==="
  bash scripts/download_kkbox.sh
else
  echo "=== [Day 1] KKBox raw files already present, skipping download ==="
fi

echo "=== [Day 1] Running Full KKBox 30-Day Baseline (Prep -> Train+Ablation -> CLV -> SHAP -> Matrix) ==="
python run_pipeline.py \
  --dataset kkbox \
  --step all \
  --seq-len 30 \
  --epochs 35 \
  --batch-size 1024 \
  --exp-name day1_baseline_30d

echo "=== [Day 1] Complete! Check outputs/kkbox/ ==="
ls -lh outputs/kkbox/
