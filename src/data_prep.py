"""Data ingestion, 30-day behavioral sequence building, feature engineering, and RFM prep."""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TrainingConfig, get_dataset_dirs

KKBOX_SEQ_CHANNELS = [
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
]


def _resolve_kkbox_file(raw_dir: Path, candidates: list[str]) -> Path:
    """Return the first existing candidate file in the KKBox raw data directory."""
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {candidates} found in {raw_dir}. Run scripts/download_kkbox.sh on the DGX first."
    )


def _prepare_kkbox(
    raw_dir: Path, seq_len: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], str, str]:
    """Load multi-million-row KKBox logs into tabular features, (N, T, C) sequences, and RFM."""
    train_path = _resolve_kkbox_file(raw_dir, ["train_v2.csv", "train.csv"])
    tx_path = _resolve_kkbox_file(raw_dir, ["transactions_v2.csv", "transactions.csv"])
    members_path = _resolve_kkbox_file(raw_dir, ["members_v3.csv", "members.csv"])
    logs_path = _resolve_kkbox_file(raw_dir, ["user_logs_v2.csv", "user_logs.csv"])

    train_lf = pl.scan_csv(train_path).select(["msno", "is_churn"])
    target_ids = train_lf.select("msno")

    tx_lf = (
        pl.scan_csv(tx_path)
        .join(target_ids, on="msno", how="semi")
        .with_columns(
            pl.col("transaction_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("tx_dt"),
            pl.col("membership_expire_date")
            .cast(pl.Utf8)
            .str.to_date("%Y%m%d", strict=False)
            .alias("exp_dt"),
            ((pl.col("actual_amount_paid") == 0) & (pl.col("plan_list_price") > 0))
            .cast(pl.Int32)
            .alias("zero_or_failed_payment"),
            (
                (pl.col("plan_list_price") - pl.col("actual_amount_paid"))
                / pl.when(pl.col("plan_list_price") > 0).then(pl.col("plan_list_price")).otherwise(1)
            ).alias("discount_pct"),
        )
    )

    tx_agg = tx_lf.group_by("msno").agg(
        pl.len().alias("tx_count"),
        pl.col("actual_amount_paid").sum().alias("total_amount_paid"),
        pl.col("actual_amount_paid").mean().alias("avg_amount_paid"),
        pl.col("payment_plan_days").mean().alias("avg_plan_days"),
        pl.col("discount_pct").mean().alias("avg_discount_pct"),
        pl.col("is_auto_renew").mean().alias("auto_renew_ratio"),
        pl.col("is_cancel").sum().alias("cancel_count"),
        pl.col("zero_or_failed_payment").sum().alias("failed_payment_count"),
        pl.col("tx_dt").min().alias("first_tx_dt"),
        pl.col("tx_dt").max().alias("last_tx_dt"),
        pl.col("exp_dt").max().alias("last_exp_dt"),
    )

    max_tx_date = tx_lf.select(pl.col("tx_dt").max()).collect().item()

    tx_features = tx_agg.with_columns(
        (pl.lit(max_tx_date) - pl.col("last_tx_dt")).dt.total_days().alias("days_since_last_tx"),
        (pl.col("last_tx_dt") - pl.col("first_tx_dt")).dt.total_days().alias("tx_recency_days"),
        (pl.lit(max_tx_date) - pl.col("first_tx_dt")).dt.total_days().alias("tx_T_days"),
    )

    logs_lf = (
        pl.scan_csv(logs_path)
        .join(target_ids, on="msno", how="semi")
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("log_dt"),
            (
                pl.col("num_25")
                + pl.col("num_50")
                + pl.col("num_75")
                + pl.col("num_985")
                + pl.col("num_100")
            ).alias("total_songs"),
        )
    )
    max_log_date = logs_lf.select(pl.col("log_dt").max()).collect().item()
    recent_7d_cutoff = max_log_date - timedelta(days=7)
    seq_start_date = max_log_date - timedelta(days=seq_len - 1)

    logs_agg = logs_lf.group_by("msno").agg(
        pl.len().alias("active_days"),
        pl.col("total_secs").sum().alias("total_secs_sum"),
        pl.col("total_secs").mean().alias("avg_daily_secs"),
        pl.col("num_unq").sum().alias("num_unq_sum"),
        (pl.col("num_100").sum() / pl.col("total_songs").sum().clip(lower_bound=1)).alias("num_100_ratio"),
        (pl.col("log_dt") >= pl.lit(recent_7d_cutoff)).sum().alias("recent_7d_active_days"),
        pl.col("total_secs").filter(pl.col("log_dt") >= pl.lit(recent_7d_cutoff)).mean().alias("recent_7d_avg_secs"),
    ).with_columns(
        (
            pl.col("recent_7d_avg_secs").fill_null(0.0)
            / pl.col("avg_daily_secs").clip(lower_bound=1.0)
        ).alias("activity_drop_ratio")
    )

    members_lf = (
        pl.scan_csv(members_path)
        .join(target_ids, on="msno", how="semi")
        .select(["msno", "city", "bd", "gender", "registered_via", "registration_init_time"])
        .with_columns(
            pl.when((pl.col("bd") >= 10) & (pl.col("bd") <= 90))
            .then(pl.col("bd"))
            .otherwise(None)
            .alias("age_clean"),
            pl.col("gender").fill_null("unknown"),
            (
                pl.lit(max_tx_date)
                - pl.col("registration_init_time").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
            )
            .dt.total_days()
            .alias("tenure_days"),
        )
        .drop(["bd", "registration_init_time"])
    )

    merged = (
        train_lf.join(tx_features, on="msno", how="left")
        .join(logs_agg, on="msno", how="left")
        .join(members_lf, on="msno", how="left")
        .drop(["first_tx_dt", "last_tx_dt", "last_exp_dt"])
        .collect(engine="streaming")
        .to_pandas()
    )

    # Build 30-day temporal behavioral tensor (N, seq_len, 7) from daily listening logs
    user_index_df = pl.DataFrame({"msno": merged["msno"].tolist()}).with_row_index("user_idx")
    seq_df = (
        logs_lf.filter(pl.col("log_dt") >= pl.lit(seq_start_date))
        .with_columns(
            (pl.col("log_dt") - pl.lit(seq_start_date)).dt.total_days().cast(pl.Int32).alias("day_idx")
        )
        .filter((pl.col("day_idx") >= 0) & (pl.col("day_idx") < seq_len))
        .group_by(["msno", "day_idx"])
        .agg([pl.col(c).sum().alias(c) for c in KKBOX_SEQ_CHANNELS])
        .join(user_index_df.lazy(), on="msno", how="inner")
        .collect(engine="streaming")
    )

    seq_tensor = np.zeros((len(merged), seq_len, len(KKBOX_SEQ_CHANNELS)), dtype=np.float32)
    if len(seq_df) > 0:
        u_idx = seq_df["user_idx"].to_numpy()
        d_idx = seq_df["day_idx"].to_numpy()
        vals = seq_df.select(KKBOX_SEQ_CHANNELS).to_numpy().astype(np.float32)
        seq_tensor[u_idx, d_idx] = np.log1p(np.clip(vals, 0.0, None))

    rfm = pd.DataFrame(
        {
            "customer_id": merged["msno"],
            "frequency": (merged["tx_count"].fillna(1) - 1).clip(lower=0),
            "recency": merged["tx_recency_days"].fillna(0).clip(lower=0),
            "T": merged["tx_T_days"].fillna(1).clip(lower=1),
            "monetary_value": merged["avg_amount_paid"].fillna(0).clip(lower=0),
        }
    )
    rfm.loc[rfm["frequency"] == 0, "recency"] = 0

    return merged, rfm, seq_tensor, KKBOX_SEQ_CHANNELS, "msno", "is_churn"


def _prepare_telco(
    raw_dir: Path, seq_len: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], str, str]:
    """Load IBM Telco Churn dataset + billing trajectory sequence proxy for smoke testing."""
    csv_path = raw_dir / "Telco-Customer-Churn.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Telco dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["is_churn"] = (df["Churn"].str.strip() == "Yes").astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["active_services_count"] = sum((df[col] == "Yes").astype(int) for col in service_cols)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["spend_change_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].clip(lower=1.0)

    tenure_months = df["tenure"].clip(lower=1)
    rfm = pd.DataFrame(
        {
            "customer_id": df["customerID"],
            "frequency": (tenure_months - 1).clip(lower=0),
            "recency": ((tenure_months - 1) * 30).clip(lower=0),
            "T": (tenure_months * 30).clip(lower=30),
            "monetary_value": df["MonthlyCharges"].clip(lower=0.01),
        }
    )

    # Construct consistent (N, seq_len, 4) trajectory proxy so the full Hybrid model can smoke-test
    seq_channels = ["monthly_spend", "avg_spend", "active_services", "contract_risk"]
    seq_tensor = np.zeros((len(df), seq_len, len(seq_channels)), dtype=np.float32)
    active_steps = np.minimum(df["tenure"].to_numpy(), seq_len)
    for i, steps in enumerate(active_steps):
        if steps > 0:
            seq_tensor[i, -steps:, 0] = np.log1p(df.loc[i, "MonthlyCharges"])
            seq_tensor[i, -steps:, 1] = np.log1p(df.loc[i, "avg_monthly_spend"])
            seq_tensor[i, -steps:, 2] = df.loc[i, "active_services_count"]
            seq_tensor[i, -steps:, 3] = df.loc[i, "is_month_to_month"]

    df = df.drop(columns=["Churn"])
    return df, rfm, seq_tensor, seq_channels, "customerID", "is_churn"


def _fit_transform_and_save(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    seq_tensor: np.ndarray,
    seq_channels: list[str],
    id_col: str,
    target_col: str,
    dataset: str,
    seed: int = 42,
) -> None:
    """Split train/val/test without leakage, normalize tabular + sequence tensors, and save."""
    dirs = get_dataset_dirs(dataset)
    proc_dir = dirs["processed"]

    y = df[target_col].astype(np.int32).to_numpy()
    ids = df[id_col].astype(str).to_numpy()
    X = df.drop(columns=[id_col, target_col])

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    for code_col in ("city", "registered_via"):
        if code_col in numeric_cols:
            numeric_cols.remove(code_col)
            categorical_cols.append(code_col)
            X[code_col] = X[code_col].astype("Int64").astype(str)

    indices = np.arange(len(df))
    idx_train_val, idx_test = train_test_split(indices, test_size=0.15, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.15 / 0.85, stratify=y[idx_train_val], random_state=seed
    )

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X.iloc[idx_train]).astype(np.float32)
    X_val = preprocessor.transform(X.iloc[idx_val]).astype(np.float32)
    X_test = preprocessor.transform(X.iloc[idx_test]).astype(np.float32)

    # Standardize sequence channels using training split statistics only
    S_train_raw = seq_tensor[idx_train]
    seq_mean = S_train_raw.mean(axis=(0, 1), keepdims=True)
    seq_std = np.where(S_train_raw.std(axis=(0, 1), keepdims=True) > 1e-6, S_train_raw.std(axis=(0, 1), keepdims=True), 1.0)

    S_train = ((S_train_raw - seq_mean) / seq_std).astype(np.float16)
    S_val = ((seq_tensor[idx_val] - seq_mean) / seq_std).astype(np.float16)
    S_test = ((seq_tensor[idx_test] - seq_mean) / seq_std).astype(np.float16)

    feature_names = preprocessor.get_feature_names_out().tolist()

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "S_train.npy", S_train)
    np.save(proc_dir / "y_train.npy", y[idx_train])

    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "S_val.npy", S_val)
    np.save(proc_dir / "y_val.npy", y[idx_val])

    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "S_test.npy", S_test)
    np.save(proc_dir / "y_test.npy", y[idx_test])

    pd.DataFrame({"customer_id": ids[idx_test], "actual_churn": y[idx_test]}).to_parquet(
        proc_dir / "test_meta.parquet", index=False
    )
    rfm.to_parquet(proc_dir / "rfm_summary.parquet", index=False)
    joblib.dump(preprocessor, proc_dir / "preprocessor.joblib")

    with open(proc_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(proc_dir / "seq_channels.json", "w", encoding="utf-8") as f:
        json.dump(seq_channels, f, indent=2)

    print(
        f"[{dataset}] Saved splits to {proc_dir}: "
        f"tabular={X_train.shape}, seq={S_train.shape}, features={len(feature_names)}"
    )


def run_data_prep(dataset: str, seq_len: int | None = None) -> None:
    """Prepare tabular features, behavioral sequence tensors, and RFM summary."""
    seq_len = seq_len or TrainingConfig().seq_len
    dirs = get_dataset_dirs(dataset)
    if dataset == "kkbox":
        df, rfm, seq_tensor, seq_channels, id_col, target_col = _prepare_kkbox(
            dirs["raw"], seq_len=seq_len
        )
    elif dataset == "telco":
        df, rfm, seq_tensor, seq_channels, id_col, target_col = _prepare_telco(
            dirs["raw"], seq_len=seq_len
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'kkbox' or 'telco'.")

    _fit_transform_and_save(df, rfm, seq_tensor, seq_channels, id_col, target_col, dataset)
