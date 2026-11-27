"""GPU-accelerated Dual-Input SHAP explainability (Tabular + 30-Day Temporal Heatmap)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from tensorflow import keras

from src.config import get_dataset_dirs
from src.model_tf import configure_gpu


def run_shap_explainability(
    dataset: str,
    background_samples: int = 256,
    explain_samples: int = 1000,
    seed: int = 42,
) -> None:
    """Compute SHAP attributions on GPU for both tabular features and 30-day sequences."""
    configure_gpu(use_mixed_precision=False)
    dirs = get_dataset_dirs(dataset)
    proc_dir = dirs["processed"]
    model_dir = dirs["models"]
    out_dir = dirs["outputs"]

    model_path = model_dir / "churn_hybrid.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run training first.")

    loaded = keras.models.load_model(model_path)
    # Wrap as ordered list inputs [tab_input, seq_input] for reliable SHAP GradientExplainer support
    tab_in = loaded.get_layer("tab_input").input
    seq_in = loaded.get_layer("seq_input").input
    wrapped_model = keras.Model(inputs=[tab_in, seq_in], outputs=loaded.output)

    X_train = np.load(proc_dir / "X_train.npy", mmap_mode="r")
    S_train = np.load(proc_dir / "S_train.npy", mmap_mode="r")
    X_test = np.load(proc_dir / "X_test.npy", mmap_mode="r")
    S_test = np.load(proc_dir / "S_test.npy", mmap_mode="r")

    with open(proc_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(proc_dir / "seq_channels.json", "r", encoding="utf-8") as f:
        seq_channels = json.load(f)

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(background_samples, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test), size=min(explain_samples, len(X_test)), replace=False)

    X_bg = np.asarray(X_train[bg_idx], dtype=np.float32)
    S_bg = np.asarray(S_train[bg_idx], dtype=np.float32)
    X_ex = np.asarray(X_test[ex_idx], dtype=np.float32)
    S_ex = np.asarray(S_test[ex_idx], dtype=np.float32)

    explainer = shap.GradientExplainer(wrapped_model, [X_bg, S_bg])
    raw_shap = explainer.shap_values([X_ex, S_ex])

    # Handle shap output shape across SHAP versions: [tab, seq] or [[tab, seq]]
    if isinstance(raw_shap, list) and len(raw_shap) == 1 and isinstance(raw_shap[0], list):
        raw_shap = raw_shap[0]

    tab_shap = np.asarray(raw_shap[0])
    seq_shap = np.asarray(raw_shap[1])
    if tab_shap.ndim == 3:
        tab_shap = tab_shap[:, :, 0]
    if seq_shap.ndim == 4:
        seq_shap = seq_shap[:, :, :, 0]

    # 1. Tabular trigger ranking
    triggers_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(tab_shap).mean(axis=0),
            "mean_signed_shap": tab_shap.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    csv_path = out_dir / "top_churn_triggers.csv"
    triggers_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(tab_shap, X_ex, feature_names=feature_names, max_display=15, show=False)
    plt.title(f"Top Tabular Churn Triggers ({dataset.upper()} - Hybrid TensorFlow SHAP)")
    plt.tight_layout()
    plot_path = out_dir / "shap_summary.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Temporal behavioral heatmap (Days before expiry x Channel)
    seq_len = seq_shap.shape[1]
    day_labels = [f"D-{seq_len - 1 - d}" for d in range(seq_len)]
    temporal_importance = np.abs(seq_shap).mean(axis=0).T  # (channels, seq_len)

    plt.figure(figsize=(14, 4.8))
    sns.heatmap(
        temporal_importance,
        xticklabels=day_labels,
        yticklabels=seq_channels,
        cmap="YlOrRd",
    )
    plt.xlabel("Days Relative to Window End (D-0 = Most Recent)")
    plt.ylabel("Behavioral Signal")
    plt.title(f"Temporal Churn Driver Heatmap ({dataset.upper()} - Mean |SHAP|)")
    plt.tight_layout()
    heatmap_path = out_dir / "shap_temporal_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    print(f"[{dataset}] Saved tabular triggers -> {csv_path}")
    print(f"[{dataset}] Saved summary plot -> {plot_path} & temporal heatmap -> {heatmap_path}")
    print(triggers_df.head(10).to_string(index=False))
