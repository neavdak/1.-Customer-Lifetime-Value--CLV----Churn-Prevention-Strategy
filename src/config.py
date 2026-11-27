"""Centralized configuration for paths, DGX GPU settings, and business assumptions."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class TrainingConfig:
    """TensorFlow training parameters tuned for DGX (32 GB RAM + 5 GB GPU)."""

    batch_size: int = 1024
    epochs: int = 35
    learning_rate: float = 1e-3
    seq_len: int = 30
    # Residual Tabular MLP + Temporal Conv1D/BiGRU/Attention capacities
    tab_hidden_units: tuple[int, ...] = (256, 256, 128)
    conv_filters: int = 64
    gru_units: int = 64
    attn_heads: int = 4
    attn_key_dim: int = 32
    fusion_units: tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.25
    l2_reg: float = 1e-4
    use_mixed_precision: bool = True
    run_ablation: bool = True
    random_seed: int = 42


@dataclass(frozen=True)
class BusinessConfig:
    """Retention strategy and financial ROI parameters."""

    clv_horizon_months: int = 12
    top_risk_fraction: float = 0.10
    retention_discount_rate: float = 0.15
    expected_retention_lift: float = 0.40
    currency_symbol: str = "₹"
    # Approximate conversion when dataset prices are in NTD/USD; set to 1.0 for native currency.
    currency_multiplier: float = 1.0


def get_dataset_dirs(dataset: str) -> dict[str, Path]:
    """Return and ensure dataset-specific processed, model, and output directories."""
    dirs = {
        "raw": DATA_RAW_DIR / ("kkbox" if dataset == "kkbox" else ""),
        "processed": DATA_PROCESSED_DIR / dataset,
        "models": MODELS_DIR / dataset,
        "outputs": OUTPUTS_DIR / dataset,
    }
    for key in ("processed", "models", "outputs"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs
