"""
lstm_model.py
-------------
Train a simple, explainable LSTM (TensorFlow/Keras) for daily sales forecasting
and compare it head-to-head against XGBoost on an IDENTICAL setup:
same store subset, same strict time-based split, same validation store-days.

Why a subset? A full per-store sequence LSTM over ~1M rows is slow and fragile on
a laptop. Restricting to the first N stores keeps the LSTM tractable and keeps the
comparison fair (XGBoost is also trained and scored only on that same subset).

How the LSTM sees the data (leakage-safe):
  For each store we slide a WINDOW-day window over the calendar. The window ending
  on day t is used to predict day t's sales. Inside that window, the model sees
  each day's sales history PLUS that day's promo/calendar — except the target
  day's own sales, which is masked to 0 so it can never see the answer. Today's
  promo and weekday ARE known in advance, so the model is allowed to use them.

INPUT:  data/features.parquet
OUTPUTS:
  outputs/metrics/lstm_metrics.json          (LSTM vs XGBoost on the same rows)
  outputs/figures/lstm_vs_xgb.png            (comparison bars)
  outputs/figures/lstm_training_curve.png    (loss per epoch)

RUN:
    pip install tensorflow         # or tensorflow-cpu (lighter, no GPU needed)
    python src/lstm_model.py
    # faster trial: LSTM_N_STORES=30 LSTM_EPOCHS=8 python src/lstm_model.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import modeling as M  # reuse FEATURE_COLS, TARGET, evaluate, train_xgb, split helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
INPUT_PATH = PROJECT_ROOT / "data" / "features.parquet"

# --- Config (override via env vars for quick trials) -----------------------
N_STORES = int(os.environ.get("LSTM_N_STORES", 100))
WINDOW = int(os.environ.get("LSTM_WINDOW", 14))     # days of history per window
EPOCHS = int(os.environ.get("LSTM_EPOCHS", 15))
BATCH = int(os.environ.get("LSTM_BATCH", 256))
LSTM_UNITS = 32
SEED = 42

# Per-timestep sequence features. Index 0 (sales) is the one we scale and mask.
SEQ_FEATURES = ["sales_scaled", "Promo", "SchoolHoliday", "dow_norm",
                "is_weekend", "Promo2Active"]
SALES_IDX = 0


def make_scaler(train_sales: np.ndarray):
    """Scale sales as standardized log1p. Fit on TRAIN sales only (no leakage).
    Returns (scale_fn, invert_fn)."""
    logs = np.log1p(np.clip(train_sales, 0, None))
    mu, sd = float(logs.mean()), float(logs.std() or 1.0)
    scale = lambda x: (np.log1p(np.clip(x, 0, None)) - mu) / sd
    invert = lambda z: np.expm1(z * sd + mu)
    return scale, invert


def build_windows(df: pd.DataFrame, cutoff: pd.Timestamp):
    """Build leakage-safe sliding windows per store.

    Returns dict of train/val arrays plus the (Store, Date) keys for validation
    so we can line the LSTM up against XGBoost on identical rows.
    """
    # Fit the sales scaler on training-period OPEN days only.
    train_mask = (df["Date"] < cutoff) & (df["Open"] == 1) & (df[M.TARGET] > 0)
    scale, invert = make_scaler(df.loc[train_mask, M.TARGET].to_numpy())

    df = df.copy()
    df["sales_scaled"] = scale(df[M.TARGET].to_numpy())
    df["dow_norm"] = (df["day_of_week"] - 1) / 6.0     # 0..1

    Xtr, ytr, Xva, yva = [], [], [], []
    va_keys, va_raw = [], []

    for store, g in df.groupby("Store", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        F = g[SEQ_FEATURES].to_numpy(dtype=np.float32)
        D = len(F)
        if D <= WINDOW:
            continue
        # windows[i] covers rows [i .. i+WINDOW-1]; its target is the last row.
        idx = np.arange(WINDOW)[None, :] + np.arange(D - WINDOW + 1)[:, None]
        windows = F[idx].copy()                         # (M, WINDOW, nfeat)
        windows[:, -1, SALES_IDX] = 0.0                 # MASK target day's sales
        tgt_pos = np.arange(WINDOW - 1, D)              # target row per window

        sales_scaled = g["sales_scaled"].to_numpy(dtype=np.float32)
        sales_raw = g[M.TARGET].to_numpy(dtype=np.float32)
        open_ = g["Open"].to_numpy()
        dates = g["Date"].to_numpy()

        for w, t in zip(windows, tgt_pos):
            if not (open_[t] == 1 and sales_raw[t] > 0):   # only real selling days
                continue
            if dates[t] < np.datetime64(cutoff):
                Xtr.append(w); ytr.append(sales_scaled[t])
            else:
                Xva.append(w); yva.append(sales_scaled[t])
                va_keys.append((store, pd.Timestamp(dates[t])))
                va_raw.append(sales_raw[t])

    return {
        "Xtr": np.asarray(Xtr, dtype=np.float32), "ytr": np.asarray(ytr, dtype=np.float32),
        "Xva": np.asarray(Xva, dtype=np.float32), "yva_raw": np.asarray(va_raw, dtype=np.float32),
        "va_keys": va_keys, "invert": invert,
    }


def build_model(n_timesteps: int, n_features: int):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input(shape=(n_timesteps, n_features)),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main() -> None:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("TensorFlow is not installed. Run:  pip install tensorflow")
        return

    np.random.seed(SEED); tf.random.set_seed(SEED)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found. Run features.py first.")

    df = pd.read_parquet(INPUT_PATH)
    stores = sorted(df["Store"].unique())[:N_STORES]
    df = df[df["Store"].isin(stores)].copy()
    cutoff = df["Date"].max() - pd.Timedelta(days=M.VALIDATION_DAYS - 1)
    print(f"Subset: {len(stores)} stores | window {WINDOW}d | cutoff {cutoff.date()}")

    # ---- LSTM ----
    data = build_windows(df, cutoff)
    print(f"LSTM windows -> train {len(data['Xtr']):,}  val {len(data['Xva']):,}")

    model = build_model(WINDOW, len(SEQ_FEATURES))
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    hist = model.fit(data["Xtr"], data["ytr"], validation_split=0.1,
                     epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=2)

    lstm_pred = data["invert"](model.predict(data["Xva"], verbose=0).ravel())
    lstm_pred = np.clip(lstm_pred, 0, None)
    lstm_df = pd.DataFrame(data["va_keys"], columns=["Store", "Date"])
    lstm_df["y_pred_lstm"] = lstm_pred
    lstm_df["Sales"] = data["yva_raw"]

    # ---- XGBoost on the SAME subset + split (fair comparison) ----
    sub = M.filter_open_positive(df)
    tr, va, _ = M.time_based_split(sub)
    xgb_model = M.train_xgb(tr)
    va = va.copy()
    va["y_pred_xgb"] = np.clip(xgb_model.predict(va[M.FEATURE_COLS]), 0, None)
    xgb_df = va[["Store", "Date", "y_pred_xgb"]]

    # ---- Align to identical (Store, Date) rows, then score both ----
    merged = lstm_df.merge(xgb_df, on=["Store", "Date"], how="inner")
    y = merged["Sales"].to_numpy()
    lstm_metrics = M.evaluate(y, merged["y_pred_lstm"].to_numpy())
    xgb_metrics = M.evaluate(y, merged["y_pred_xgb"].to_numpy())

    print("\n" + "=" * 56)
    print(f"LSTM vs XGBoost  (identical {len(merged):,} validation store-days)")
    print("=" * 56)
    print(f"{'metric':<8}{'LSTM':>14}{'XGBoost':>14}")
    for m in ["MAE", "RMSE", "SMAPE", "MAPE"]:
        print(f"{m:<8}{lstm_metrics[m]:>14.2f}{xgb_metrics[m]:>14.2f}")
    winner = "XGBoost" if xgb_metrics["MAE"] < lstm_metrics["MAE"] else "LSTM"
    print(f"\nLower MAE: {winner}")

    # ---- Save metrics + figures ----
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump({"n_stores": len(stores), "window": WINDOW, "n_val_rows": int(len(merged)),
               "lstm": lstm_metrics, "xgboost_subset": xgb_metrics},
              open(OUT / "metrics" / "lstm_metrics.json", "w"), indent=2)

    # comparison bars
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, keys, title in [(a1, ["MAE", "RMSE"], "Error in sales units"),
                            (a2, ["SMAPE", "MAPE"], "Percentage error (%)")]:
        x = np.arange(len(keys)); w = 0.35
        ax.bar(x - w/2, [lstm_metrics[k] for k in keys], w, label="LSTM", color="#9c6ade")
        ax.bar(x + w/2, [xgb_metrics[k] for k in keys], w, label="XGBoost", color="#e08a1e")
        ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_title(title); ax.legend()
    fig.suptitle(f"LSTM vs XGBoost on {len(stores)} stores (lower is better)")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "lstm_vs_xgb.png", dpi=120); plt.close(fig)

    # training curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.history["loss"], label="train loss", color="#3b6ea5")
    if "val_loss" in hist.history:
        ax.plot(hist.history["val_loss"], label="val loss", color="#e08a1e")
    ax.set_title("LSTM training curve"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figures" / "lstm_training_curve.png", dpi=120); plt.close(fig)

    print(f"\nSaved lstm_metrics.json, lstm_vs_xgb.png, lstm_training_curve.png -> outputs/")
    print("Done.")


if __name__ == "__main__":
    main()