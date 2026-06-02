"""
inventory.py
------------
Translate the sales FORECASTS into INVENTORY DECISIONS.

Core formula (per store, per day):
    recommended_order_quantity = predicted_demand + safety_stock - current_inventory

- predicted_demand  : the XGBoost forecast from modeling.py.
- safety_stock      : a buffer for forecast uncertainty, sized from each store's
                      recent forecast error:  safety_stock = z * recent_error_std.
- current_inventory : SIMULATED here (Rossmann has no real inventory data). It is
                      clearly labelled 'sim_' and exists only to demonstrate the
                      decision logic end to end.

Each row also gets a stockout RISK FLAG (low / medium / high).

INPUT:  outputs/predictions/validation_predictions.parquet  (from modeling.py)
OUTPUTS:
  outputs/predictions/inventory_recommendations.parquet   (all validation rows)
  outputs/predictions/inventory_sample.csv                (a few stores, readable)

RUN:    python src/inventory.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
PRED_PATH = OUT / "predictions" / "validation_predictions.parquet"

# --- Config ----------------------------------------------------------------
SERVICE_LEVEL = 0.95          # target: cover demand ~95% of the time
Z = float(norm.ppf(SERVICE_LEVEL))   # z-score for that service level (~1.645)
SIM_SEED = 42                 # makes the simulated inventory reproducible
SAMPLE_STORES = 5             # how many stores to write to the readable CSV


def add_safety_stock(df: pd.DataFrame) -> pd.DataFrame:
    """safety_stock = Z * (this store's recent forecast-error std).

    Forecast error = actual - predicted. A store whose sales are hard to predict
    (large error std) gets a bigger buffer; a steady, predictable store gets a
    smaller one. That is the whole idea of error-based safety stock: protect more
    where uncertainty is higher. Z encodes the service level we're targeting."""
    df = df.copy()
    df["forecast_error"] = df["Sales"] - df["y_pred_xgb"]
    err_std = df.groupby("Store")["forecast_error"].transform(lambda s: s.std(ddof=1))
    # Stores with too few points to estimate a std fall back to the global median.
    err_std = err_std.fillna(err_std.median())
    df["recent_error_std"] = err_std
    df["safety_stock"] = (Z * df["recent_error_std"]).round().clip(lower=0)
    return df


def simulate_current_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Create a SIMULATED on-hand inventory level (Rossmann has none).

    We draw each store-day's stock as a random fraction (0.85x-1.30x) of
    predicted demand, centred near typical real stock levels so the demo
    naturally contains a balanced mix of understocked, balanced, and overstocked
    situations to exercise the risk logic. Clearly prefixed 'sim_' so no one
    mistakes it for real data. (Widen/shift this band to taste — it only changes
    the simulated scenario, nothing in the forecasting or order logic.)"""
    rng = np.random.default_rng(SIM_SEED)
    factor = rng.uniform(0.85, 1.30, size=len(df))
    df = df.copy()
    df["sim_current_inventory"] = (df["y_pred_xgb"] * factor).round().clip(lower=0)
    return df


def recommend(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the order formula and assign a stockout risk flag.

    Risk compares simulated stock on hand against demand + buffer:
      - high   : stock won't even cover expected demand        -> likely stockout
      - medium : covers demand but not the safety buffer       -> exposed
      - low    : covers demand AND the safety buffer            -> comfortable
    The recommended order quantity refills up to (demand + safety stock), never
    negative (you can't un-order stock)."""
    df = df.copy()
    df["predicted_demand"] = df["y_pred_xgb"].round().clip(lower=0)
    target_level = df["predicted_demand"] + df["safety_stock"]

    df["recommended_order_qty"] = (
        (target_level - df["sim_current_inventory"]).round().clip(lower=0)
    )

    conditions = [
        df["sim_current_inventory"] < df["predicted_demand"],
        df["sim_current_inventory"] < target_level,
    ]
    df["risk_flag"] = np.select(conditions, ["high", "medium"], default="low")
    return df


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"{PRED_PATH} not found. Run src/modeling.py first to create predictions."
        )

    preds = pd.read_parquet(PRED_PATH)
    preds = add_safety_stock(preds)
    preds = simulate_current_inventory(preds)
    preds = recommend(preds)

    cols = ["Store", "Date", "predicted_demand", "safety_stock",
            "sim_current_inventory", "recommended_order_qty", "risk_flag"]
    table = preds[cols].sort_values(["Store", "Date"]).reset_index(drop=True)

    # Save the full recommendation table.
    full_path = OUT / "predictions" / "inventory_recommendations.parquet"
    table.to_parquet(full_path, index=False)

    # Save a small, human-readable sample (first few stores).
    sample_stores = sorted(table["Store"].unique())[:SAMPLE_STORES]
    sample = table[table["Store"].isin(sample_stores)].copy()
    sample_path = OUT / "predictions" / "inventory_sample.csv"
    sample.to_csv(sample_path, index=False)

    # Console summary.
    print("=" * 70)
    print(f"INVENTORY RECOMMENDATIONS  (service level {SERVICE_LEVEL:.0%}, Z={Z:.3f})")
    print("=" * 70)
    print("\nNOTE: 'sim_current_inventory' is SIMULATED for demonstration only.\n")
    print("Sample (one store, first rows):")
    one = table[table["Store"] == sample_stores[0]].head(7)
    print(one.to_string(index=False))

    counts = table["risk_flag"].value_counts()
    total = len(table)
    print("\nStockout risk across all validation store-days:")
    for flag in ["low", "medium", "high"]:
        n = int(counts.get(flag, 0))
        print(f"  {flag:<7}: {n:>7,}  ({n/total:5.1%})")

    print(f"\nSaved -> {full_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved -> {sample_path.relative_to(PROJECT_ROOT)}")
    print("Done. Next step: visualization (src/visualization.py).")


if __name__ == "__main__":
    main()