# Project Summary
### Retail Sales Forecasting & Inventory Optimization

**Goal.** Forecast daily store sales and convert those forecasts into inventory
order recommendations. A portfolio prototype / decision-support project — not a
production system; the inventory layer uses simulated on-hand stock.

**Data.** Rossmann Store Sales (Kaggle): 1,017,209 daily records across 1,115
stores, Jan 2013 – Jul 2015. Validation came from a strict time-based split, not
a random one.

**Approach.**
- Cleaned and merged sales with store metadata; modeled only open days with
  positive sales.
- Engineered calendar, promotion, and recent-sales-history features
  (lags + rolling stats), all shifted to prevent future-data leakage; excluded
  `Customers` (unknown at forecast time).
- Compared a 7-day-average **baseline** against an **XGBoost** model, held out
  the final 6 weeks (2015-06-20 → 2015-07-31). Also trained an **LSTM**
  (TensorFlow/Keras) and compared it fairly against XGBoost on a 100-store subset.
- Added an inventory layer: error-based safety stock (95% service level) →
  recommended order quantity → low/medium/high stockout-risk flag. Plus a **PuLP
  linear program** for cost-minimizing order quantities (holding + stockout cost
  subject to lead-time, storage, and safety-stock constraints; rolling-horizon).
- Tracked experiments with MLflow (optional script).

**Results (validation, last 6 weeks).**

| Metric | Baseline (7-day avg) | XGBoost |
|---|---:|---:|
| MAE | 1597.34 | **579.05** |
| RMSE | 2323.40 | **840.64** |
| SMAPE | 23.78% | **8.50%** |
| MAPE | 22.04% | **8.74%** |

XGBoost reduced MAE by **~63.7%** over the baseline — roughly 8–9% average error
on unseen weeks. Top drivers: recent sales history (`sales_lag_14`,
`rolling_mean_28`, `sales_lag_28`) and the `Promo` flag. Promotions lift average
sales by **38.8%**. Simulated stockout-risk mix at 95% service level: ~27% low /
~40% medium / ~33% high. In a controlled comparison on 100 stores, an LSTM was
competitive (9.50% SMAPE) but XGBoost was more accurate (8.28%) and far faster.
The PuLP optimization (safety stock + lead time + capacity, rolling-horizon) cut
realized cost by ~3–5% versus a simple order-up-to policy under illustrative
cost assumptions.

**Limitations.** Simulated inventory; safety stock estimated on the validation
window; rolling features include closed-day zeros; single global model with
default-ish parameters and no tuning yet.

**Next steps.** Real inventory + lead-time data; rolling trailing error for
safety stock; quantile forecasts; tuning + log-transformed target; multi-window
backtesting; a deeper / tuned LSTM on all stores to test if it closes the gap.

**Repo.** Code in `src/`, narrative in `notebooks/01_retail_sales_forecasting.ipynb`,
full write-up in `README.md`, presentation/interview material in
`reports/explanation_materials.md`.