"""
visualization.py
-----------------
Produce ALL project charts in one place and save them as PNGs under
outputs/figures/. Charts read from saved artifacts, so nothing is retrained.

Three groups:
  EDA (from data/clean_merged.parquet):
    eda_sales_over_time, eda_sales_by_dow, eda_sales_by_month,
    eda_sales_by_storetype, eda_promo_effect, eda_store_variability
  EVALUATION (from outputs/predictions/validation_predictions.parquet + metrics.json):
    eval_actual_vs_pred_scatter, eval_daily_actual_vs_pred,
    eval_model_comparison, eval_error_distribution, eval_store_forecast
  INVENTORY (from outputs/predictions/inventory_recommendations.parquet):
    inv_order_by_store, inv_demand_vs_stock, inv_risk_breakdown

Each chart is wrapped in try/except in main(), so a missing input skips just
that chart instead of stopping the whole run.

RUN: python src/visualization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "outputs"
FIG = OUT / "figures"

CLEAN_PATH = DATA_DIR / "clean_merged.parquet"
PRED_PATH = OUT / "predictions" / "validation_predictions.parquet"
INV_PATH = OUT / "predictions" / "inventory_recommendations.parquet"
METRICS_PATH = OUT / "metrics" / "metrics.json"

BLUE, ORANGE, GREEN, RED = "#3b6ea5", "#e08a1e", "#3a9c6b", "#c0504d"
RISK_COLORS = {"low": GREEN, "medium": ORANGE, "high": RED}


def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    plt.close(fig)
    print(f"  saved figures/{name}")


def _open_positive(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Open"] == 1) & (df["Sales"] > 0)]


# ----------------------------- EDA -----------------------------------------
def eda_sales_over_time(df):
    d = _open_positive(df)
    daily = d.groupby("Date")["Sales"].mean()
    roll = daily.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.values, color=BLUE, alpha=0.35, label="Daily avg")
    ax.plot(roll.index, roll.values, color=BLUE, lw=2, label="7-day rolling avg")
    ax.set_title("Average daily sales per open store over time")
    ax.set_ylabel("Sales"); ax.legend()
    _save(fig, "eda_sales_over_time.png")


def eda_sales_by_dow(df):
    d = _open_positive(df)
    order = [1, 2, 3, 4, 5, 6, 7]
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    means = d.groupby("DayOfWeek")["Sales"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, means.values, color=BLUE)
    ax.set_title("Average sales by day of week (open stores)")
    ax.set_ylabel("Avg sales")
    _save(fig, "eda_sales_by_dow.png")


def eda_sales_by_month(df):
    d = _open_positive(df).copy()
    d["month"] = d["Date"].dt.month
    means = d.groupby("month")["Sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(means.index, means.values, marker="o", color=BLUE)
    ax.set_xticks(range(1, 13))
    ax.set_title("Average sales by month (seasonality)")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg sales")
    _save(fig, "eda_sales_by_month.png")


def eda_sales_by_storetype(df):
    d = _open_positive(df)
    means = d.groupby("StoreType")["Sales"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(means.index.astype(str), means.values, color=BLUE)
    ax.set_title("Average sales by store type")
    ax.set_xlabel("StoreType"); ax.set_ylabel("Avg sales")
    _save(fig, "eda_sales_by_storetype.png")


def eda_promo_effect(df):
    d = _open_positive(df)
    means = d.groupby("Promo")["Sales"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No promo", "Promo"], [means.get(0, 0), means.get(1, 0)],
           color=[BLUE, ORANGE])
    ax.set_title("Average sales: promotion vs no promotion")
    ax.set_ylabel("Avg sales")
    _save(fig, "eda_promo_effect.png")


def eda_store_variability(df):
    d = _open_positive(df)
    per_store = d.groupby("Store")["Sales"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_store.values, bins=40, color=BLUE, edgecolor="white")
    ax.set_title("Store-level variability: distribution of each store's avg sales")
    ax.set_xlabel("Average daily sales (per store)"); ax.set_ylabel("Number of stores")
    _save(fig, "eda_store_variability.png")


# -------------------------- EVALUATION -------------------------------------
def eval_actual_vs_pred_scatter(preds):
    s = preds.sample(min(5000, len(preds)), random_state=0)
    lim = max(s["Sales"].max(), s["y_pred_xgb"].max())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s["Sales"], s["y_pred_xgb"], s=6, alpha=0.25, color=BLUE)
    ax.plot([0, lim], [0, lim], color=RED, lw=1.5, label="perfect")
    ax.set_title("Actual vs predicted sales (validation)")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.legend()
    _save(fig, "eval_actual_vs_pred_scatter.png")


def eval_daily_actual_vs_pred(preds):
    daily = preds.groupby("Date")[["Sales", "y_pred_xgb"]].mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily["Sales"], color=BLUE, lw=2, label="Actual")
    ax.plot(daily.index, daily["y_pred_xgb"], color=ORANGE, lw=2, ls="--", label="Predicted")
    ax.set_title("Daily average actual vs predicted (validation window)")
    ax.set_ylabel("Avg sales"); ax.legend()
    _save(fig, "eval_daily_actual_vs_pred.png")


def eval_model_comparison(metrics):
    base, xgbm = metrics["baseline"], metrics["xgboost"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, keys, title in [(a1, ["MAE", "RMSE"], "Error in sales units"),
                            (a2, ["SMAPE", "MAPE"], "Percentage error (%)")]:
        x = np.arange(len(keys)); w = 0.35
        ax.bar(x - w/2, [base[k] for k in keys], w, label="Baseline", color=BLUE)
        ax.bar(x + w/2, [xgbm[k] for k in keys], w, label="XGBoost", color=ORANGE)
        ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_title(title); ax.legend()
    fig.suptitle("Model comparison: baseline vs XGBoost (lower is better)")
    _save(fig, "eval_model_comparison.png")


def eval_error_distribution(preds):
    resid = preds["Sales"] - preds["y_pred_xgb"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(resid, bins=60, color=BLUE, edgecolor="white")
    ax.axvline(0, color=RED, lw=1.5)
    ax.set_title("Forecast error distribution (actual − predicted)")
    ax.set_xlabel("Error"); ax.set_ylabel("Count")
    _save(fig, "eval_error_distribution.png")


def eval_store_forecast(preds):
    store = preds["Store"].value_counts().idxmax()  # a well-populated store
    s = preds[preds["Store"] == store].sort_values("Date")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s["Date"], s["Sales"], color=BLUE, marker="o", ms=3, label="Actual")
    ax.plot(s["Date"], s["y_pred_xgb"], color=ORANGE, marker="o", ms=3, ls="--", label="Predicted")
    ax.set_title(f"Store {store}: actual vs predicted over validation weeks")
    ax.set_ylabel("Sales"); ax.legend()
    _save(fig, "eval_store_forecast.png")


# -------------------------- INVENTORY --------------------------------------
def inv_order_by_store(inv):
    top = (inv.groupby("Store")["recommended_order_qty"].mean()
           .sort_values(ascending=False).head(20).iloc[::-1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index.astype(str), top.values, color=BLUE)
    ax.set_title("Avg recommended order quantity — top 20 stores")
    ax.set_xlabel("Avg recommended order qty"); ax.set_ylabel("Store")
    _save(fig, "inv_order_by_store.png")


def inv_demand_vs_stock(inv):
    s = inv.sample(min(4000, len(inv)), random_state=0).copy()
    s["target_level"] = s["predicted_demand"] + s["safety_stock"]
    lim = max(s["sim_current_inventory"].max(), s["target_level"].max())
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for flag, c in RISK_COLORS.items():
        m = s["risk_flag"] == flag
        ax.scatter(s.loc[m, "sim_current_inventory"], s.loc[m, "target_level"],
                   s=8, alpha=0.4, color=c, label=flag)
    ax.plot([0, lim], [0, lim], color="gray", lw=1, ls=":")
    ax.set_title("Stock on hand vs required (demand + safety)\npoints above line need ordering")
    ax.set_xlabel("Simulated current inventory"); ax.set_ylabel("Required stock")
    ax.legend(title="risk")
    _save(fig, "inv_demand_vs_stock.png")


def inv_risk_breakdown(inv):
    counts = inv["risk_flag"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[RISK_COLORS[k] for k in counts.index])
    ax.set_title("Stockout risk flags across validation store-days")
    ax.set_ylabel("Count")
    _save(fig, "inv_risk_breakdown.png")


# ----------------------------- main ----------------------------------------
def main():
    print("Generating charts -> outputs/figures/")

    def run(fn, *args):
        try:
            fn(*args)
        except FileNotFoundError as e:
            print(f"  skip {fn.__name__}: missing input ({e})")
        except Exception as e:  # keep going if one chart fails
            print(f"  skip {fn.__name__}: {e}")

    if CLEAN_PATH.exists():
        df = pd.read_parquet(CLEAN_PATH)
        for fn in [eda_sales_over_time, eda_sales_by_dow, eda_sales_by_month,
                   eda_sales_by_storetype, eda_promo_effect, eda_store_variability]:
            run(fn, df)
    else:
        print(f"  skip EDA charts: {CLEAN_PATH} not found (run data_prep.py)")

    if PRED_PATH.exists():
        preds = pd.read_parquet(PRED_PATH)
        for fn in [eval_actual_vs_pred_scatter, eval_daily_actual_vs_pred,
                   eval_error_distribution, eval_store_forecast]:
            run(fn, preds)
    else:
        print(f"  skip evaluation charts: {PRED_PATH} not found (run modeling.py)")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            run(eval_model_comparison, json.load(f))
    else:
        print(f"  skip model comparison: {METRICS_PATH} not found")

    if INV_PATH.exists():
        inv = pd.read_parquet(INV_PATH)
        for fn in [inv_order_by_store, inv_demand_vs_stock, inv_risk_breakdown]:
            run(fn, inv)
    else:
        print(f"  skip inventory charts: {INV_PATH} not found (run inventory.py)")

    print("Done. Next step: notebook + MLflow + README.")


if __name__ == "__main__":
    main()