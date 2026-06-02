"""
data_prep.py
------------
Load, inspect, clean, and merge the Rossmann Store Sales data into a single
time-sorted table that the feature-engineering step can build on.

INPUTS (you place these in the data/ folder yourself — see data/README.md):
    data/train.csv   daily sales per store (contains the Sales target)
    data/store.csv   one row per store with store-level metadata

OUTPUT:
    data/clean_merged.parquet   cleaned, merged, time-sorted dataset

RUN:
    python src/data_prep.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------
# Resolve everything relative to the project root (the folder above src/) so
# the script works no matter which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
STORE_PATH = DATA_DIR / "store.csv"
OUTPUT_PATH = DATA_DIR / "clean_merged.parquet"


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw CSV files.

    Two deliberate choices here:
    - parse_dates=["Date"] makes Date a real datetime instead of text, so we
      can sort and extract calendar features later.
    - StateHoliday is read as a string because the original file mixes the
      integer 0 and the string "0" in the same column, which otherwise makes
      pandas guess an awkward 'object' dtype and complicate comparisons.
    """
    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["Date"],
        dtype={"StateHoliday": "str"},
        low_memory=False,
    )
    store = pd.read_csv(STORE_PATH)
    return train, store


def inspect(train: pd.DataFrame, store: pd.DataFrame) -> None:
    """Print a quick data-quality report. This only reads data, never changes it."""
    line = "=" * 60
    print(line)
    print("RAW DATA INSPECTION")
    print(line)

    print(f"\ntrain.csv shape : {train.shape}")
    print(f"store.csv shape : {store.shape}")

    print("\ntrain columns / dtypes:")
    print(train.dtypes)

    print("\nDate range:")
    print(f"  {train['Date'].min().date()}  ->  {train['Date'].max().date()}")
    print(f"  unique days   : {train['Date'].nunique()}")
    print(f"  unique stores : {train['Store'].nunique()}")

    miss_train = train.isna().sum()
    print("\nMissing values in train:")
    print(miss_train[miss_train > 0] if miss_train.sum() else "  none")

    miss_store = store.isna().sum()
    print("\nMissing values in store:")
    print(miss_store[miss_store > 0] if miss_store.sum() else "  none")

    # Business sanity checks — these are the numbers we discuss in the EDA.
    closed = int((train["Open"] == 0).sum())
    zero_sales = int((train["Sales"] == 0).sum())
    zero_sales_open = int(((train["Sales"] == 0) & (train["Open"] == 1)).sum())
    promo_days = int((train["Promo"] == 1).sum())

    print("\nBusiness checks:")
    print(f"  closed-day rows (Open==0)   : {closed:,}")
    print(f"  zero-sales rows (Sales==0)  : {zero_sales:,}")
    print(f"  zero sales while OPEN       : {zero_sales_open:,}")
    print(f"  promo days (Promo==1)       : {promo_days:,}")
    print("\n  StateHoliday value counts:")
    print(train["StateHoliday"].value_counts().to_string())
    print("\n  StoreType value counts:")
    print(store["StoreType"].value_counts().to_string())
    print("\n  Assortment value counts:")
    print(store["Assortment"].value_counts().to_string())
    print(line)


def clean_store(store: pd.DataFrame) -> pd.DataFrame:
    """Fill missing store metadata with explicit, documented choices.

    The store table has three kinds of missingness:
    - CompetitionDistance: missing for a few stores -> fill with the median
      distance (a neutral 'typical' value) so we keep those stores.
    - CompetitionOpenSince* : unknown competitor open date -> fill with 0 as an
      explicit 'unknown / not applicable' marker the model can read as a signal.
    - Promo2Since* / PromoInterval: missing exactly when the store never runs
      the ongoing 'Promo2' campaign -> fill with 0 / "None".
    """
    store = store.copy()

    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(
        store["CompetitionDistance"].median()
    )

    for col in [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        store[col] = store[col].fillna(0).astype(int)

    store["PromoInterval"] = store["PromoInterval"].fillna("None")
    return store


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    """Tidy the transactional table.

    Normalise StateHoliday so the 'no holiday' value is consistently the string
    "0" (the raw file mixes 0 and "0"). Values 'a'/'b'/'c' mean public/Easter/
    Christmas holidays respectively.
    """
    train = train.copy()
    train["StateHoliday"] = train["StateHoliday"].replace({"0.0": "0"}).astype(str)
    train["StateHoliday"] = train["StateHoliday"].replace({"0": "0"})  # no-op safeguard
    return train


def merge_and_sort(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Attach store metadata to every transaction row and sort by time.

    Sorting by (Store, Date) is critical: the lag and rolling features we build
    in the next step rely on rows being in chronological order *within each
    store*. If rows were out of order, a shift() would silently mix stores or
    jump around in time and create data leakage.

    NOTE on closed days: we keep them here on purpose. They carry the calendar
    so that 'sales 7 days ago' lines up with the same weekday last week. The
    decision to *train and evaluate only on open days* is applied later, at the
    modeling stage, and is documented there.
    """
    df = train.merge(store, on="Store", how="left")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def main() -> None:
    train, store = load_raw()
    inspect(train, store)

    store = clean_store(store)
    train = clean_train(train)
    df = merge_and_sort(train, store)

    print(f"\nMerged dataset shape          : {df.shape}")
    print(f"Remaining missing values total: {int(df.isna().sum().sum())}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved clean merged data -> {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Done. Next step: feature engineering (src/features.py).")


if __name__ == "__main__":
    main()
