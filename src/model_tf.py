"""DGX GPU-accelerated Hybrid Sequence (Conv1D + BiGRU + Self-Attention) + Tabular ResNet model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import time
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.config import TrainingConfig, get_dataset_dirs


def configure_gpu(use_mixed_precision: bool = True) -> list[tf.config.PhysicalDevice]:
    """Enable GPU memory growth (safe for 5 GB VRAM) and Tensor Core mixed_float16."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus and use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    return gpus


def _make_hybrid_dataset(
    X: np.ndarray,
    S: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Stream mini-batches from mmap/NumPy arrays without exceeding 5 GB GPU VRAM."""
    n = len(y)
    tab_dim = X.shape[1]
    seq_len, seq_ch = S.shape[1], S.shape[2]

    def gen() -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            yield (
                {
                    "tab_input": np.asarray(X[batch_idx], dtype=np.float32),
                    "seq_input": np.asarray(S[batch_idx], dtype=np.float32),
                },
                np.asarray(y[batch_idx], dtype=np.float32),
            )

    output_signature = (
        {
            "tab_input": tf.TensorSpec(shape=(None, tab_dim), dtype=tf.float32),
            "seq_input": tf.TensorSpec(shape=(None, seq_len, seq_ch), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).prefetch(
        tf.data.AUTOTUNE
    )


def _tabular_resnet_block(
    x: tf.Tensor, units: int, dropout_rate: float, l2_reg: float
) -> tf.Tensor:
    """Residual dense block for tabular transaction & profile features."""
    shortcut = x if x.shape[-1] == units else layers.Dense(units, use_bias=False)(x)
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("swish")(h)
    h = layers.Dropout(dropout_rate)(h)
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(h)
    h = layers.BatchNormalization()(h)
    return layers.Activation("swish")(layers.Add()([shortcut, h]))


def build_tabular_baseline(tab_dim: int, seq_shape: tuple[int, int], cfg: TrainingConfig) -> keras.Model:
    """Baseline Tabular ResNet model (ignores temporal sequence) for DGX ablation."""
    tab_in = keras.Input(shape=(tab_dim,), name="tab_input")
    # Accept seq_input so both models share the exact same tf.data pipeline
    _seq_in = keras.Input(shape=seq_shape, name="seq_input")

    x = tab_in
    for units in cfg.tab_hidden_units:
        x = _tabular_resnet_block(x, units, cfg.dropout_rate, cfg.l2_reg)

    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="churn_prob")(x)
    model = keras.Model(inputs={"tab_input": tab_in, "seq_input": _seq_in}, outputs=outputs, name="tabular_resnet_baseline")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def build_hybrid_seq_attn_net(
    tab_dim: int, seq_shape: tuple[int, int], cfg: TrainingConfig
) -> keras.Model:
    """Hybrid Conv1D + BiGRU + Multi-Head Self-Attention + Tabular ResNet churn model."""
    tab_in = keras.Input(shape=(tab_dim,), name="tab_input")
    seq_in = keras.Input(shape=seq_shape, name="seq_input")

    # 1. Temporal Behavioral Sequence Branch (30-day daily activity logs)
    s = layers.Conv1D(
        filters=cfg.conv_filters,
        kernel_size=3,
        padding="causal",
        activation="swish",
        kernel_regularizer=regularizers.l2(cfg.l2_reg),
    )(seq_in)
    s = layers.LayerNormalization()(s)
    s = layers.Bidirectional(
        layers.GRU(cfg.gru_units, return_sequences=True, dropout=0.15)
    )(s)

    attn_out = layers.MultiHeadAttention(
        num_heads=cfg.attn_heads, key_dim=cfg.attn_key_dim, dropout=0.15
    )(s, s)
    s = layers.LayerNormalization()(layers.Add()([s, attn_out]))

    seq_avg = layers.GlobalAveragePooling1D()(s)
    seq_max = layers.GlobalMaxPooling1D()(s)
    seq_repr = layers.Concatenate()([seq_avg, seq_max])

    # 2. Tabular ResNet Branch (demographics, contracts, payment friction, RFM stats)
    t = tab_in
    for units in cfg.tab_hidden_units:
        t = _tabular_resnet_block(t, units, cfg.dropout_rate, cfg.l2_reg)

    # 3. Fusion Head
    fused = layers.Concatenate()([t, seq_repr])
    for units in cfg.fusion_units:
        fused = layers.Dense(units, kernel_regularizer=regularizers.l2(cfg.l2_reg), use_bias=False)(fused)
        fused = layers.BatchNormalization()(fused)
        fused = layers.Activation("swish")(fused)
        fused = layers.Dropout(cfg.dropout_rate)(fused)

    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="churn_prob")(fused)
    model = keras.Model(
        inputs={"tab_input": tab_in, "seq_input": seq_in},
        outputs=outputs,
        name="hybrid_seq_attn_net",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=1e-4),
        loss=keras.losses.BinaryFocalCrossentropy(gamma=1.5, apply_class_balancing=False),
        metrics=[
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def _calibrate_probs(
    val_probs: np.ndarray, y_val: np.ndarray, test_probs: np.ndarray
) -> np.ndarray:
    """Platt-calibrate probabilities on validation log-odds for accurate revenue-at-risk."""
    eps = 1e-6
    val_logits = np.log(np.clip(val_probs, eps, 1 - eps) / (1 - np.clip(val_probs, eps, 1 - eps)))
    test_logits = np.log(np.clip(test_probs, eps, 1 - eps) / (1 - np.clip(test_probs, eps, 1 - eps)))

    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(val_logits.reshape(-1, 1), y_val)
    return calibrator.predict_proba(test_logits.reshape(-1, 1))[:, 1]


def _evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute classification, ranking, and calibration metrics."""
    top_10_k = max(1, int(math.ceil(len(y_true) * 0.10)))
    top_10_idx = np.argsort(probs)[::-1][:top_10_k]
    preds = (probs >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "precision_at_top10pct": float(y_true[top_10_idx].mean()),
        "f1_at_0_5": float(f1_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
    }


def _train_single_model(
    model: keras.Model,
    model_path,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    steps_per_epoch: int,
    val_steps: int,
    class_weight: dict[int, float],
    epochs: int,
) -> keras.Model:
    """Fit a Keras model with checkpointing, early stopping, and LR scheduling."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_roc_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return model


def _append_experiment_log(
    out_dir,
    experiment_name: str,
    model_name: str,
    seq_len: int,
    cfg: TrainingConfig,
    train_seconds: float,
    metrics: dict[str, float],
) -> None:
    """Append run hyperparameters, runtime, and metrics to experiment_history.csv."""
    log_path = out_dir / "experiment_history.csv"
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_name": experiment_name,
        "model_name": model_name,
        "seq_len": seq_len,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "dropout_rate": cfg.dropout_rate,
        "conv_filters": cfg.conv_filters,
        "gru_units": cfg.gru_units,
        "attn_heads": cfg.attn_heads,
        "train_seconds": round(train_seconds, 2),
        **metrics,
    }
    row_df = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        pd.concat([existing, row_df], ignore_index=True).to_csv(log_path, index=False)
    else:
        row_df.to_csv(log_path, index=False)


def train_churn_model(
    dataset: str,
    cfg: TrainingConfig | None = None,
    experiment_name: str = "default",
) -> None:
    """Train Hybrid Sequence+Tabular Net (and optional Baseline ResNet ablation) on DGX."""
    cfg = cfg or TrainingConfig()
    tf.keras.utils.set_random_seed(cfg.random_seed)
    gpus = configure_gpu(cfg.use_mixed_precision)
    print(f"[{dataset}] TensorFlow devices detected: GPUs={len(gpus)}")

    dirs = get_dataset_dirs(dataset)
    proc_dir = dirs["processed"]
    model_dir = dirs["models"]
    out_dir = dirs["outputs"]

    X_train = np.load(proc_dir / "X_train.npy", mmap_mode="r")
    S_train = np.load(proc_dir / "S_train.npy", mmap_mode="r")
    y_train = np.load(proc_dir / "y_train.npy")

    X_val = np.load(proc_dir / "X_val.npy", mmap_mode="r")
    S_val = np.load(proc_dir / "S_val.npy", mmap_mode="r")
    y_val = np.load(proc_dir / "y_val.npy")

    X_test = np.load(proc_dir / "X_test.npy", mmap_mode="r")
    S_test = np.load(proc_dir / "S_test.npy", mmap_mode="r")
    y_test = np.load(proc_dir / "y_test.npy")
    test_meta = pd.read_parquet(proc_dir / "test_meta.parquet")

    train_ds = _make_hybrid_dataset(X_train, S_train, y_train, cfg.batch_size, shuffle=True, seed=cfg.random_seed).repeat()
    val_ds = _make_hybrid_dataset(X_val, S_val, y_val, cfg.batch_size, shuffle=False).repeat()
    val_eval_ds = _make_hybrid_dataset(X_val, S_val, y_val, cfg.batch_size, shuffle=False)
    test_eval_ds = _make_hybrid_dataset(X_test, S_test, y_test, cfg.batch_size, shuffle=False)

    steps_per_epoch = math.ceil(len(y_train) / cfg.batch_size)
    val_steps = math.ceil(len(y_val) / cfg.batch_size)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    tab_dim = X_train.shape[1]
    seq_shape = (S_train.shape[1], S_train.shape[2])

    results: dict[str, dict[str, float]] = {}
    curve_data: dict[str, np.ndarray] = {}

    if cfg.run_ablation:
        print(f"[{dataset}] Training Baseline Tabular ResNet for ablation...")
        t0 = time.perf_counter()
        base_model = build_tabular_baseline(tab_dim, seq_shape, cfg)
        base_model = _train_single_model(
            base_model,
            model_dir / "baseline_tabular.keras",
            train_ds,
            val_ds,
            steps_per_epoch,
            val_steps,
            class_weight,
            cfg.epochs,
        )
        base_seconds = time.perf_counter() - t0
        base_val = base_model.predict(val_eval_ds, verbose=0).ravel()
        base_test_raw = base_model.predict(test_eval_ds, verbose=0).ravel()
        base_test_cal = _calibrate_probs(base_val, y_val, base_test_raw)
        results["baseline_tabular_resnet"] = _evaluate_predictions(y_test, base_test_cal)
        curve_data["Baseline Tabular ResNet"] = base_test_cal
        _append_experiment_log(
            out_dir,
            experiment_name,
            "baseline_tabular_resnet",
            seq_shape[0],
            cfg,
            base_seconds,
            results["baseline_tabular_resnet"],
        )

    print(f"[{dataset}] Training Hybrid Conv1D + BiGRU + Self-Attention + Tabular ResNet...")
    t0 = time.perf_counter()
    hybrid_model = build_hybrid_seq_attn_net(tab_dim, seq_shape, cfg)
    hybrid_path = model_dir / "churn_hybrid.keras"
    hybrid_model = _train_single_model(
        hybrid_model,
        hybrid_path,
        train_ds,
        val_ds,
        steps_per_epoch,
        val_steps,
        class_weight,
        cfg.epochs,
    )
    hybrid_seconds = time.perf_counter() - t0

    val_raw = hybrid_model.predict(val_eval_ds, verbose=0).ravel()
    test_raw = hybrid_model.predict(test_eval_ds, verbose=0).ravel()
    test_calibrated = _calibrate_probs(val_raw, y_val, test_raw)

    results["hybrid_seq_attn_net"] = _evaluate_predictions(y_test, test_calibrated)
    curve_data["Hybrid Seq+Attn+ResNet"] = test_calibrated
    _append_experiment_log(
        out_dir,
        experiment_name,
        "hybrid_seq_attn_net",
        seq_shape[0],
        cfg,
        hybrid_seconds,
        results["hybrid_seq_attn_net"],
    )

    with open(out_dir / "model_ablation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    test_meta["churn_prob"] = test_calibrated
    test_meta["churn_prob_uncalibrated"] = test_raw
    test_meta.to_parquet(model_dir / "test_predictions.parquet", index=False)

    # Save ROC & PR comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, probs in curve_data.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, probs):.4f})")
        axes[1].plot(rec, prec, label=f"{label} (AP={average_precision_score(y_test, probs):.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC Curve Comparison")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall Curve Comparison")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=200)
    plt.close()

    print(f"[{dataset}] Saved Hybrid model to {hybrid_path}")
    print(json.dumps(results, indent=2))
