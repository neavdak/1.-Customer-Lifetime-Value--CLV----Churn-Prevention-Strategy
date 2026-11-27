"""Executive Deliverable: 2x2 Prioritization Matrix, Top 10% ROI & Targeting Curve."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BusinessConfig, get_dataset_dirs


def run_prioritization_matrix(dataset: str, biz_cfg: BusinessConfig | None = None) -> None:
    """Combine churn probabilities with 12-month CLV to build the 2x2 matrix and ROI curve."""
    biz_cfg = biz_cfg or BusinessConfig()
    dirs = get_dataset_dirs(dataset)
    model_dir = dirs["models"]
    out_dir = dirs["outputs"]

    preds_path = model_dir / "test_predictions.parquet"
    clv_path = out_dir / "clv_predictions.parquet"
    if not preds_path.exists() or not clv_path.exists():
        raise FileNotFoundError(
            "Both test_predictions.parquet and clv_predictions.parquet are required before building the matrix."
        )

    preds = pd.read_parquet(preds_path)
    clv = pd.read_parquet(clv_path)
    df = preds.merge(clv, on="customer_id", how="inner")

    df["expected_revenue_at_risk"] = df["churn_prob"] * df["predicted_clv_12m"]

    clv_cutoff = float(df["predicted_clv_12m"].median())
    risk_cutoff = float(df["churn_prob"].median())

    df["clv_tier"] = np.where(df["predicted_clv_12m"] >= clv_cutoff, "High CLV", "Low CLV")
    df["risk_tier"] = np.where(df["churn_prob"] >= risk_cutoff, "High Churn Risk", "Low Churn Risk")
    df["quadrant"] = df["clv_tier"] + " / " + df["risk_tier"]

    ranked_by_value_risk = df.sort_values("expected_revenue_at_risk", ascending=False)
    ranked_by_churn = df.sort_values("churn_prob", ascending=False)

    n_target = max(1, int(np.ceil(len(df) * biz_cfg.top_risk_fraction)))
    top_risk_df = ranked_by_churn.head(n_target).copy()
    top_10pct_value_targets = ranked_by_value_risk.head(n_target).copy()
    high_priority = df[df["quadrant"] == "High CLV / High Churn Risk"].sort_values(
        "expected_revenue_at_risk", ascending=False
    )

    def _roi_metrics(subset: pd.DataFrame) -> dict[str, float | int]:
        rev_at_risk = float(subset["expected_revenue_at_risk"].sum())
        discount_cost = float((subset["predicted_clv_12m"] * biz_cfg.retention_discount_rate).sum())
        gross_saved = rev_at_risk * biz_cfg.expected_retention_lift
        net_saved = gross_saved - discount_cost
        return {
            "accounts": int(len(subset)),
            "total_clv_12m": float(subset["predicted_clv_12m"].sum()),
            "expected_revenue_at_risk": rev_at_risk,
            "retention_discount_cost": discount_cost,
            "gross_revenue_saved": float(gross_saved),
            "net_revenue_saved": float(net_saved),
        }

    # Evaluate ROI sensitivity across targeting depths (5% to 50%)
    curve_rows = []
    for pct in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        k = max(1, int(np.ceil(len(df) * pct)))
        m = _roi_metrics(ranked_by_value_risk.head(k))
        curve_rows.append({"target_pct": pct * 100, **m})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_dir / "roi_sensitivity.csv", index=False)

    optimal_row = curve_df.loc[curve_df["net_revenue_saved"].idxmax()]

    summary = {
        "dataset": dataset,
        "currency": biz_cfg.currency_symbol,
        "total_test_accounts": int(len(df)),
        "clv_median_cutoff": clv_cutoff,
        "churn_prob_median_cutoff": risk_cutoff,
        "assumptions": {
            "top_risk_fraction": biz_cfg.top_risk_fraction,
            "retention_discount_rate": biz_cfg.retention_discount_rate,
            "expected_retention_lift": biz_cfg.expected_retention_lift,
        },
        "top_10pct_highest_churn_prob": _roi_metrics(top_risk_df),
        "top_10pct_highest_revenue_at_risk": _roi_metrics(top_10pct_value_targets),
        "high_clv_high_risk_quadrant": _roi_metrics(high_priority),
        "optimal_targeting_pct": float(optimal_row["target_pct"]),
        "optimal_net_revenue_saved": float(optimal_row["net_revenue_saved"]),
        "quadrant_counts": df["quadrant"].value_counts().to_dict(),
    }

    summary_path = out_dir / "executive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    targets_path = out_dir / "top_10pct_targets.csv"
    top_10pct_value_targets.to_csv(targets_path, index=False)

    # 1. 2x2 Prioritization Matrix scatter plot
    plot_df = df.sample(n=min(len(df), 25_000), random_state=42)
    palette = {
        "High CLV / High Churn Risk": "#d62728",
        "High CLV / Low Churn Risk": "#2ca02c",
        "Low CLV / High Churn Risk": "#ff7f0e",
        "Low CLV / Low Churn Risk": "#7f7f7f",
    }

    plt.figure(figsize=(10, 7))
    for quadrant, color in palette.items():
        sub = plot_df[plot_df["quadrant"] == quadrant]
        plt.scatter(
            sub["churn_prob"],
            sub["predicted_clv_12m"],
            c=color,
            label=f"{quadrant} (n={summary['quadrant_counts'].get(quadrant, 0):,})",
            alpha=0.55,
            s=18,
        )

    plt.axvline(risk_cutoff, color="black", linestyle="--", linewidth=1.2)
    plt.axhline(clv_cutoff, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("Calibrated Next-Month Churn Probability")
    plt.ylabel(f"Predicted 12-Month CLV ({biz_cfg.currency_symbol})")
    plt.title(
        f"2x2 Retention Prioritization Matrix ({dataset.upper()}) | "
        f"Top 10% Net Saved: {biz_cfg.currency_symbol}{summary['top_10pct_highest_revenue_at_risk']['net_revenue_saved']:,.2f}"
    )
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plot_path = out_dir / "prioritization_matrix.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # 2. ROI Sensitivity Curve (Top 5% -> 50%)
    plt.figure(figsize=(9, 5))
    plt.plot(curve_df["target_pct"], curve_df["gross_revenue_saved"], marker="o", label="Gross Revenue Saved")
    plt.plot(curve_df["target_pct"], curve_df["retention_discount_cost"], marker="s", label="Discount Cost")
    plt.plot(curve_df["target_pct"], curve_df["net_revenue_saved"], marker="^", linewidth=2.2, label="Net Revenue Saved")
    plt.axvline(10, color="gray", linestyle="--", label="Top 10% Target")
    plt.xlabel("Targeted At-Risk Customer Share (%)")
    plt.ylabel(f"Amount ({biz_cfg.currency_symbol})")
    plt.title(f"Retention Campaign ROI Curve ({dataset.upper()})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    roi_plot_path = out_dir / "roi_sensitivity_curve.png"
    plt.savefig(roi_plot_path, dpi=200)
    plt.close()

    print(f"[{dataset}] Saved matrix -> {plot_path}, ROI curve -> {roi_plot_path}, summary -> {summary_path}")
    print(json.dumps(summary, indent=2))
