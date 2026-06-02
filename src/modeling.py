"""
modeling.py
-----------
Train and evaluate the sales forecasting models on the engineered feature table.

Pipeline:
  1. Load features.parquet.
  2. Keep only OPEN days with POSITIVE sales (the real forecasting task).
  3. Strict TIME-BASED split: hold out the final 6 weeks for validation.
  4. Baseline   : predict the previous-7-day rolling average (a business rule).
  5. Main model : XGBoost Regressor on the engineered features.
  6. Evaluate   : MAE, RMSE, SMAPE, MAPE on the validation weeks.
  7. Save       : metrics JSON, validation predictions, feature-importance plot,
                  and the trained model.

INPUT:  data/features.parquet   (from src/features.py)
OUTPUTS:
  outputs/metrics/metrics.json
  outputs/predictions/validation_predictions.parquet
  outputs/figures/feature_importance.png
  outputs/xgb_model.json

RUN:    python src/modeling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: save files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "outputs"
INPUT_PATH = DATA_DIR / "features.parquet"

# --- Config ----------------------------------------------------------------
VALIDATION_WEEKS = 6
VALIDATION_DAYS = VALIDATION_WEEKS * 7

# The exact columns fed to XGBoost. Defined explicitly (not "everything except")
# so it is obvious what the model sees — and so we can DELIBERATELY EXCLUDE
# leakage traps. Most important exclusion: 'Customers'. The customer count is
# only known AFTER a day happens, so it is not available when forecasting future
# sales; using it would leak the answer. We also drop the raw string categoricals
# (we use their *_enc versions) and 'Open' (constant after filtering to open days).
FEATURE_COLS = [
    # store identity + metadata
    "Store",
    "StoreType_enc", "Assortment_enc",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "Promo2Active",
    # calendar
    "year", "month", "week", "day", "day_of_week",
    "is_weekend", "is_month_start", "is_month_end",
    # promotions / holidays for the day
    "Promo", "SchoolHoliday", "StateHoliday_enc",
    # leakage-safe history
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28",
    "store_expanding_mean",
]
TARGET = "Sales"

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    tree_method="hist",   # fast histogram algorithm; fine on a laptop
    random_state=42,
    n_jobs=-1,
)


# --- Metrics (computed from data — never hard-coded) -----------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true, y_pred):
    """Symmetric MAPE in %. Symmetric => over- and under-forecasts are treated
    even-handedly, and it stays bounded (0-200%) even when actuals are small."""
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.where(denom == 0, 0.0, 2.0 * np.abs(y_pred - y_true) / denom)
    return float(np.mean(diff) * 100.0)


def mape(y_true, y_pred):
    """Plain MAPE in %. Valid here because we evaluate only on Sales > 0 rows."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def evaluate(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "SMAPE": smape(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


# --- Data prep for modeling ------------------------------------------------
def filter_open_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only real selling days: store open AND sales > 0.

    Closed days are zero by definition, so leaving them in lets a model 'win' by
    predicting an obvious rule instead of forecasting demand — which would
    dishonestly flatter every metric. The 54 open-but-zero rows are dropped as
    data-quality noise."""
    return df[(df["Open"] == 1) & (df[TARGET] > 0)].copy()


def time_based_split(df: pd.DataFrame):
    """Split by DATE, not at random. The last VALIDATION_DAYS become validation;
    everything earlier is training.

    Forecasting means predicting the future from the past. A random split would
    scatter future dates into the training set, letting the model peek ahead and
    earn scores it could never repeat in production. A time split mimics reality:
    train on the past, score on a future window the model has not seen."""
    cutoff = df["Date"].max() - pd.Timedelta(days=VALIDATION_DAYS - 1)
    train = df[df["Date"] < cutoff].copy()
    valid = df[df["Date"] >= cutoff].copy()
    return train, valid, cutoff


# --- Models ----------------------------------------------------------------
def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """Business-rule baseline: predict today's sales as the previous 7-day
    average (the leakage-safe rolling_mean_7 we already built). Every ML model
    must beat this to justify its complexity."""
    return df["rolling_mean_7"].to_numpy()


def train_xgb(train: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train[FEATURE_COLS], train[TARGET])
    return model


def save_feature_importance(model: xgb.XGBRegressor, path: Path, top_n: int = 20):
    """Plot the top features by GAIN (how much each feature improved the model's
    splits) — more meaningful than raw split counts."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")  # {feature_name: gain}
    imp = (pd.Series(gain).sort_values(ascending=False).head(top_n).iloc[::-1])

    plt.figure(figsize=(8, max(4, 0.4 * len(imp))))
    plt.barh(imp.index, imp.values, color="#3b6ea5")
    plt.xlabel("Importance (gain)")
    plt.title(f"XGBoost — Top {len(imp)} Features by Gain")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    df = filter_open_positive(df)
    train, valid, cutoff = time_based_split(df)

    print("=" * 60)
    print("TIME-BASED SPLIT")
    print("=" * 60)
    print(f"Validation window : last {VALIDATION_WEEKS} weeks "
          f"({cutoff.date()} -> {df['Date'].max().date()})")
    print(f"Train rows        : {len(train):,}")
    print(f"Valid rows        : {len(valid):,}")

    y_valid = valid[TARGET].to_numpy()

    # Baseline
    base_pred = baseline_predict(valid)
    base_metrics = evaluate(y_valid, base_pred)

    # XGBoost
    print("\nTraining XGBoost ...")
    model = train_xgb(train)
    xgb_pred = model.predict(valid[FEATURE_COLS])
    xgb_pred = np.clip(xgb_pred, 0, None)  # sales can't be negative
    xgb_metrics = evaluate(y_valid, xgb_pred)

    # Report
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (lower is better)")
    print("=" * 60)
    print(f"{'metric':<8}{'Baseline(7d avg)':>20}{'XGBoost':>14}")
    for m in ["MAE", "RMSE", "SMAPE", "MAPE"]:
        print(f"{m:<8}{base_metrics[m]:>20.2f}{xgb_metrics[m]:>14.2f}")

    improve = (base_metrics["MAE"] - xgb_metrics["MAE"]) / base_metrics["MAE"] * 100
    verdict = "beats" if xgb_metrics["MAE"] < base_metrics["MAE"] else "does NOT beat"
    print(f"\nXGBoost {verdict} the baseline on MAE by {improve:.1f}%.")

    # Save artifacts
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "metrics.json", "w") as f:
        json.dump({"validation_window_days": VALIDATION_DAYS,
                   "n_train": len(train), "n_valid": len(valid),
                   "baseline": base_metrics, "xgboost": xgb_metrics}, f, indent=2)

    preds = valid[["Store", "Date", TARGET]].copy()
    preds["y_pred_baseline"] = base_pred
    preds["y_pred_xgb"] = xgb_pred
    preds.to_parquet(OUT / "predictions" / "validation_predictions.parquet", index=False)

    save_feature_importance(model, OUT / "figures" / "feature_importance.png")
    model.save_model(OUT / "xgb_model.json")

    print(f"\nSaved metrics, predictions, feature_importance.png, and xgb_model.json -> outputs/")
    print("Done. Next step: inventory layer (src/inventory.py).")


if __name__ == "__main__":
    main()