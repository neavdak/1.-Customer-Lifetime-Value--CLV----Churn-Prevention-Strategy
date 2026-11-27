"""CLI entry point for the CLV & Churn Prevention pipeline (run on college DGX)."""

from __future__ import annotations

import argparse

from src.config import BusinessConfig, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Lifetime Value (CLV) & Churn Prevention Pipeline (Hybrid TensorFlow + Lifetimes)"
    )
    parser.add_argument(
        "--dataset",
        choices=["kkbox", "telco"],
        default="kkbox",
        help="Target dataset ('kkbox' for multi-million-row DGX run, 'telco' for quick test).",
    )
    parser.add_argument(
        "--step",
        choices=["all", "prep", "train", "clv", "shap", "matrix"],
        default="all",
        help="Pipeline stage to execute.",
    )
    # Data & Model Hyperparameters (Day 1 & Day 2 DGX sweeps)
    parser.add_argument("--seq-len", type=int, default=30, help="Daily sequence window length (e.g., 30, 60, 90).")
    parser.add_argument("--epochs", type=int, default=35, help="TensorFlow training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for 5 GB GPU VRAM.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--conv-filters", type=int, default=64, help="Conv1D filters in sequence branch.")
    parser.add_argument("--gru-units", type=int, default=64, help="BiGRU units per direction.")
    parser.add_argument("--attn-heads", type=int, default=4, help="MultiHeadAttention heads.")
    parser.add_argument("--exp-name", type=str, default="default", help="Label logged to experiment_history.csv.")
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="Skip training the Baseline Tabular ResNet and train only the Hybrid Sequence+Attention model.",
    )
    # SHAP parameters
    parser.add_argument("--shap-bg", type=int, default=256, help="SHAP background sample count.")
    parser.add_argument("--shap-samples", type=int, default=1000, help="SHAP explanation sample count.")
    # Business / ROI calibration parameters (Day 3)
    parser.add_argument("--currency", type=str, default="₹", help="Currency symbol for executive charts.")
    parser.add_argument("--currency-multiplier", type=float, default=1.0, help="Multiplier for CLV currency conversion.")
    parser.add_argument("--discount-rate", type=float, default=0.15, help="Retention offer discount fraction.")
    parser.add_argument("--retention-lift", type=float, default=0.40, help="Expected fraction of churners saved.")
    args = parser.parse_args()

    biz_cfg = BusinessConfig(
        retention_discount_rate=args.discount_rate,
        expected_retention_lift=args.retention_lift,
        currency_symbol=args.currency,
        currency_multiplier=args.currency_multiplier,
    )

    if args.step in ("all", "prep"):
        from src.data_prep import run_data_prep

        run_data_prep(args.dataset, seq_len=args.seq_len)

    if args.step in ("all", "train"):
        from src.model_tf import train_churn_model

        cfg = TrainingConfig(
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dropout_rate=args.dropout,
            conv_filters=args.conv_filters,
            gru_units=args.gru_units,
            attn_heads=args.attn_heads,
            run_ablation=not args.no_ablation,
        )
        train_churn_model(args.dataset, cfg=cfg, experiment_name=args.exp_name)

    if args.step in ("all", "clv"):
        from src.clv_model import run_clv_model

        run_clv_model(args.dataset, biz_cfg=biz_cfg)

    if args.step in ("all", "shap"):
        from src.explain_shap import run_shap_explainability

        run_shap_explainability(
            args.dataset,
            background_samples=args.shap_bg,
            explain_samples=args.shap_samples,
        )

    if args.step in ("all", "matrix"):
        from src.prioritization_matrix import run_prioritization_matrix

        run_prioritization_matrix(args.dataset, biz_cfg=biz_cfg)


if __name__ == "__main__":
    main()
