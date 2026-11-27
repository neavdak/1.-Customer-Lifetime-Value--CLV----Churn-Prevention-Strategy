"""12-month Customer Lifetime Value (CLV) estimation using BG/NBD + Gamma-Gamma."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

from src.config import BusinessConfig, get_dataset_dirs


def run_clv_model(dataset: str, biz_cfg: BusinessConfig | None = None) -> None:
    """Fit BG/NBD and Gamma-Gamma models on RFM summary and predict 12-month CLV."""
    biz_cfg = biz_cfg or BusinessConfig()
    dirs = get_dataset_dirs(dataset)
    proc_dir = dirs["processed"]
    out_dir = dirs["outputs"]

    rfm = pd.read_parquet(proc_dir / "rfm_summary.parquet")

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])

    # Expected transactions over the next 12 months (~365 days; recency/T are in days)
    rfm["pred_purchases_12m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        365.0, rfm["frequency"], rfm["recency"], rfm["T"]
    )

    repeat_mask = (rfm["frequency"] > 0) & (rfm["monetary_value"] > 0)
    if repeat_mask.sum() >= 50:
        ggf = GammaGammaFitter(penalizer_coef=0.01)
        ggf.fit(rfm.loc[repeat_mask, "frequency"], rfm.loc[repeat_mask, "monetary_value"])

        clv_repeat = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[repeat_mask, "frequency"],
            rfm.loc[repeat_mask, "recency"],
            rfm.loc[repeat_mask, "T"],
            rfm.loc[repeat_mask, "monetary_value"],
            time=biz_cfg.clv_horizon_months,
            discount_rate=0.01,
            freq="D",
        )
        rfm["predicted_clv_12m"] = 0.0
        rfm.loc[repeat_mask, "predicted_clv_12m"] = clv_repeat.values

        # Fallback for single-purchase customers (frequency == 0)
        non_repeat = ~repeat_mask
        rfm.loc[non_repeat, "predicted_clv_12m"] = (
            rfm.loc[non_repeat, "pred_purchases_12m"] * rfm.loc[non_repeat, "monetary_value"]
        )
    else:
        rfm["predicted_clv_12m"] = rfm["pred_purchases_12m"] * rfm["monetary_value"]

    rfm["predicted_clv_12m"] = (
        rfm["predicted_clv_12m"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        * biz_cfg.currency_multiplier
    )

    out_path = out_dir / "clv_predictions.parquet"
    rfm[["customer_id", "pred_purchases_12m", "predicted_clv_12m"]].to_parquet(
        out_path, index=False
    )
    print(
        f"[{dataset}] Saved 12-month CLV predictions for {len(rfm)} customers to {out_path} "
        f"(mean CLV = {biz_cfg.currency_symbol}{rfm['predicted_clv_12m'].mean():.2f})"
    )
