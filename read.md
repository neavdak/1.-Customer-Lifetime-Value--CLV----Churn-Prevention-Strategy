# Customer Lifetime Value (CLV) & Churn Prevention Strategy

* **Business Problem:** Acquiring a new customer costs 5x more than retaining an existing one. Management needs to know who will churn next month, how much revenue is at risk, and which customers are worth offering a retention discount.

* **What to Run on DGX (32 GB RAM + 5 GB GPU) — Hybrid TensorFlow Stack:**
  * **Data & Sequence Ingestion (32 GB RAM):** Load multi-million-row behavioral logs (KKBox `user_logs_v2.csv`, `transactions_v2.csv`, `members_v3.csv`) into 32 GB RAM using **Polars** streaming, extracting both aggregated tabular features and a **30-day daily behavioral sequence tensor `(N, 30, 7)`** per customer, streamed from memory-mapped `.npy` files via `tf.data.Dataset` so 5 GB GPU VRAM is never exceeded.
  * **Hybrid Deep Churn + CLV Modeling (GPU TensorFlow + Lifetimes):**
    * Train a **Hybrid Sequence + Tabular Deep Net** (`Conv1D + Bidirectional GRU + Multi-Head Self-Attention` on the 30-day listening logs fused with a `Tabular ResNet` on contract/payment features), using **Binary Focal Loss**, `mixed_float16` Tensor Cores, **Platt probability calibration**, and an automated **Baseline Tabular ResNet vs. Hybrid Sequence Net ablation** (`model_comparison.png`).
    * Fit **BG/NBD + Gamma-Gamma (`lifetimes`)** for predicted 12-month CLV.
  * **Dual-Input Explainability (GPU):** Run **SHAP (`GradientExplainer`)** on the GPU across both inputs to produce ranked tabular churn triggers (`shap_summary.png`) **and** a **30-day Temporal Churn Heatmap** (`shap_temporal_heatmap.png`).

* **Executive Deliverable:** A 2x2 prioritization matrix (High CLV / High Churn Risk) + **ROI Sensitivity Curve** (`5%`–`50%` targeting depth) showing exact ₹/$ revenue saved when targeting the top 10% at-risk accounts.

---

## Project Requirements & Stack

* **Hardware:** DGX / workstation with 32 GB RAM + 5 GB CUDA-enabled GPU
* **Dataset:** KKBox Churn Prediction (recommended for multi-million-row behavioral logs) or Telco / Bank Customer Churn datasets
* **Core Python Libraries:**
  * `tensorflow` (with GPU/CUDA support), `keras`
  * `polars` or `pandas`, `numpy`
  * `lifetimes` (BG/NBD & Gamma-Gamma for 12-month CLV)
  * `shap` (`GradientExplainer` / `DeepExplainer`)
  * `scikit-learn` (preprocessing, scaling, encoding, train/test split, metrics)
  * `matplotlib` / `seaborn` / `plotly` (2x2 matrix & SHAP visualizations)

---

## College Project To-Do Checklist

1. **Environment Setup**
   * Set up Python environment with CUDA/cuDNN compatible `tensorflow` and verify GPU detection (`tf.config.list_physical_devices('GPU')`).
   * Install `polars`, `pandas`, `scikit-learn`, `lifetimes`, `shap`, and plotting libraries.
2. **Dataset Acquisition & Exploration**
   * Download the dataset (e.g., KKBox or Telco/Bank Churn) and inspect schemas, row counts, class balance, and target churn definition.
3. **Data Loading & Feature Engineering (32 GB RAM)**
   * Load logs with Polars/Pandas and engineer features (RFM metrics, login/usage frequency trends, failed payments, tenure).
   * Normalize numerical features, encode categoricals (or use Keras embedding/preprocessing layers), and perform a leak-free train/validation/test split.
   * Build a `tf.data.Dataset` pipeline with batching and prefetching for the 5 GB GPU.
4. **Train TensorFlow Churn Model (GPU)**
   * Build and train a Keras MLP (or LSTM/GRU if using time-series logs) with binary cross-entropy, class weighting/focal loss for imbalance, and early stopping.
   * Evaluate with ROC-AUC, PR-AUC, F1-score, and probability calibration.
5. **Predict 12-Month CLV**
   * Prepare RFM summary (`frequency`, `recency`, `T`, `monetary_value`) and fit the **BG/NBD** model (`lifetimes`) to estimate 12-month CLV per customer.
6. **Explain Churn Triggers (GPU)**
   * Run SHAP (`GradientExplainer` / `DeepExplainer`) on the TensorFlow model to rank top churn drivers globally and per customer.
7. **Build Executive 2x2 Matrix & ROI Summary**
   * Plot **Predicted Churn Probability** vs. **Predicted 12-Month CLV** into a 2x2 prioritization matrix.
   * Identify the **High CLV / High Churn Risk** segment (top 10% at-risk accounts) and compute total revenue at risk and expected ₹/$ revenue saved after retention discount costs.

---

## Project Structure & How to Run at College (DGX)

```text
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv   # Already downloaded for quick testing
├── scripts/
│   ├── download_kkbox.sh              # Downloads full multi-million-row KKBox dataset
│   ├── day1_setup_and_baseline.sh     # Day 1: venv, Telco smoke test, KKBox 30d baseline
│   ├── day2_dgx_experiments.sh        # Day 2: 60d sequence scaling, tuning & 5k-sample SHAP
│   └── day3_finalize_and_package.sh   # Day 3: final ROI calibration & tar.gz backup
├── src/
│   ├── config.py                      # DGX GPU & business ROI config
│   ├── data_prep.py                   # Polars streaming + 30/60-day sequence tensors + RFM
│   ├── model_tf.py                    # Hybrid Conv1D+BiGRU+Attention+ResNet + experiment log
│   ├── clv_model.py                   # BG/NBD + Gamma-Gamma 12-month CLV
│   ├── explain_shap.py                # Dual-input GPU SHAP + temporal heatmap
│   └── prioritization_matrix.py       # 2x2 matrix, top 10% targets & ROI curve
├── requirements.txt
└── run_pipeline.py                    # Single CLI runner (supports --seq-len, --lr, etc.)
```

### 3-Day College DGX Execution Plan

```bash
# Day 1: Setup env, smoke-test Telco, download KKBox (~/.kaggle/kaggle.json required), run 30-day baseline
bash scripts/day1_setup_and_baseline.sh

# Day 2: Run hyperparameter sweeps, 60-day sequence scaling & 5,000-sample GPU SHAP
bash scripts/day2_dgx_experiments.sh

# Day 3: Calibrate ROI currency/discount assumptions & package outputs/models into dgx_project_deliverables.tar.gz
bash scripts/day3_finalize_and_package.sh
```

Or run individual stages manually with `run_pipeline.py`:
```bash
python run_pipeline.py --dataset kkbox --step all --seq-len 30 --epochs 35 --batch-size 1024
```
