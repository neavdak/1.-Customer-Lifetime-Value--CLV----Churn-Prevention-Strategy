#!/usr/bin/env bash
set -euo pipefail

# Day 3 on College DGX: Final business ROI calibration & deliverables backup package.
# Optional overrides: CURRENCY=₹ CURRENCY_MULTIPLIER=1.0 DISCOUNT_RATE=0.15 RETENTION_LIFT=0.40

source .venv/bin/activate

CURRENCY="${CURRENCY:-₹}"
CURRENCY_MULTIPLIER="${CURRENCY_MULTIPLIER:-1.0}"
DISCOUNT_RATE="${DISCOUNT_RATE:-0.15}"
RETENTION_LIFT="${RETENTION_LIFT:-0.40}"

echo "=== [Day 3] Finalizing 2x2 Matrix & ROI Curve ==="
python run_pipeline.py \
  --dataset kkbox \
  --step matrix \
  --currency "$CURRENCY" \
  --currency-multiplier "$CURRENCY_MULTIPLIER" \
  --discount-rate "$DISCOUNT_RATE" \
  --retention-lift "$RETENTION_LIFT"

echo "=== [Day 3] Packaging Deliverables Archive (excluding raw multi-GB CSVs) ==="
ARCHIVE_NAME="dgx_project_deliverables.tar.gz"
tar -czf "$ARCHIVE_NAME" \
  outputs/ \
  models/ \
  read.md \
  requirements.txt \
  run_pipeline.py \
  src/ \
  scripts/

echo "=== [Day 3] Done! Copy '$ARCHIVE_NAME' back to your MacBook before DGX access ends ==="
ls -lh "$ARCHIVE_NAME"
