"""
features.py
-----------
Turn the cleaned, merged table from data_prep.py into a model-ready feature
table. Adds calendar features, encoded categoricals, a Promo2-active flag, and —
most importantly — LEAKAGE-SAFE lag and rolling features, where every row only
ever sees data from strictly *before* its own date.

INPUT:  data/clean_merged.parquet   (produced by src/data_prep.py)
OUTPUT: data/features.parquet

RUN:    python src/features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "clean_merged.parquet"
OUTPUT_PATH = DATA_DIR / "features.parquet"

# --- Config ----------------------------------------------------------------
LAGS = [7, 14, 28]        # "what were sales N days ago"
WINDOWS = [7, 14, 28]     # rolling-window sizes in days

# Deterministic category -> integer maps. We define them by hand (instead of
# sklearn's LabelEncoder) so the encoding is fixed, documented, and identical
# every run. These categories are nominal; tree models like XGBoost handle
# integer-coded nominal features well, and they learn feature interactions
# automatically — which is why we don't hand-build many interaction columns.
STORETYPE_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_MAP = {"a": 0, "b": 1, "c": 2}
STATEHOLIDAY_MAP = {"0": 0, "a": 1, "b": 2, "c": 3}
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar signals derived purely from the Date — no leakage possible."""
    d = df["Date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["week"] = d.isocalendar().week.astype(int)
    df["day"] = d.day
    df["day_of_week"] = d.dayofweek + 1            # 1=Mon ... 7=Sun (Rossmann style)
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_month_end"] = d.is_month_end.astype(int)
    return df


def add_categorical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Map store metadata + holiday flags to fixed integer codes."""
    df["StoreType_enc"] = df["StoreType"].map(STORETYPE_MAP).astype(int)
    df["Assortment_enc"] = df["Assortment"].map(ASSORTMENT_MAP).astype(int)
    df["StateHoliday_enc"] = (
        df["StateHoliday"].astype(str).map(STATEHOLIDAY_MAP).fillna(0).astype(int)
    )
    return df


def add_promo2_active(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where the store's ongoing 'Promo2' campaign is active today.

    Active requires all three: the store participates (Promo2 == 1); today is on
    or after the campaign's start (year + ISO week); and the current month is one
    of the store's PromoInterval months (e.g. "Jan,Apr,Jul,Oct"). Uses only
    same-row metadata, so no leakage.
    """
    month_name = df["month"].map(MONTH_ABBR)
    started = (
        (df["year"] > df["Promo2SinceYear"])
        | ((df["year"] == df["Promo2SinceYear"]) & (df["week"] >= df["Promo2SinceWeek"]))
    )
    in_interval = pd.Series(
        [m in str(pi).split(",") for m, pi in zip(month_name, df["PromoInterval"])],
        index=df.index,
    )
    df["Promo2Active"] = ((df["Promo2"] == 1) & started & in_interval).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """sales_lag_N = that store's Sales exactly N calendar days earlier.

    groupby('Store').shift(N) shifts each store's series down by N rows. Because
    rows are sorted by (Store, Date) and the calendar is complete (closed days
    kept), shifting by 7 lands on the same weekday one week back. The shift only
    ever pulls values from earlier rows, so a row can never see its own future.
    """
    g = df.groupby("Store")["Sales"]
    for lag in LAGS:
        df[f"sales_lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std over the days strictly *before* each row.

    The leakage guard is shift(1) applied BEFORE rolling: it pushes the window
    back by one day so the current day's Sales (our target) is excluded.
    rolling_mean_7 at a given row = average of that store's previous 7 days.
    Using .transform on the groupby keeps every window inside a single store, so
    windows never spill across store boundaries.
    """
    g = df.groupby("Store")["Sales"]
    for w in WINDOWS:
        df[f"rolling_mean_{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=2).std()
        )
    # Expanding mean of all prior days = a leakage-safe "store historical average"
    # that grows as more history accumulates.
    df["store_expanding_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    return df


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    # Re-sort defensively; every shift/rolling op below depends on this order.
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_categorical_encodings(df)
    df = add_promo2_active(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop warm-up rows that lack full lag history (the first ~28 days per
    # store). Every remaining row has complete lag features.
    before = len(df)
    df = df.dropna(subset=[f"sales_lag_{max(LAGS)}"]).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} warm-up rows lacking full lag history.")

    # rolling_std needs >=2 points; fill the few remaining NaNs with 0
    # (no variation observed yet).
    std_cols = [c for c in df.columns if c.startswith("rolling_std_")]
    df[std_cols] = df[std_cols].fillna(0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Feature table shape : {df.shape}")
    print(f"Remaining NaNs      : {int(df.isna().sum().sum())}")
    print(f"Saved features -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Done. Next step: modeling (src/modeling.py).")


if __name__ == "__main__":
    main()