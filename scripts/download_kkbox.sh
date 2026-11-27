#!/usr/bin/env bash
set -euo pipefail

# Download KKBox Churn Prediction Challenge dataset on the college DGX.
# Prerequisite: place your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json)
# and accept the competition rules on Kaggle.

TARGET_DIR="data/raw/kkbox"
mkdir -p "$TARGET_DIR"

echo "Downloading KKBox dataset into $TARGET_DIR..."
kaggle competitions download -c kkbox-churn-prediction-challenge -p "$TARGET_DIR"

echo "Extracting archives..."
for archive in "$TARGET_DIR"/*.zip "$TARGET_DIR"/*.7z; do
  [ -e "$archive" ] || continue
  case "$archive" in
    *.zip) unzip -o "$archive" -d "$TARGET_DIR" ;;
    *.7z)  7z x "$archive" -o"$TARGET_DIR" -y ;;
  esac
done

echo "Done. Files in $TARGET_DIR:"
ls -lh "$TARGET_DIR"
