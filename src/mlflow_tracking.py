"""
mlflow_tracking.py  (OPTIONAL)
------------------------------
Re-run the baseline + XGBoost training INSIDE an MLflow run so the experiment is
tracked: parameters, metrics, the chart artifacts, and the trained model are all
logged and browsable in the MLflow UI.

This is deliberately a SEPARATE, optional script. The core pipeline
(data_prep -> features -> modeling -> inventory -> visualization) does NOT depend
on MLflow, so if MLflow is not installed or you don't need tracking, nothing else
breaks. This script imports the modeling logic instead of duplicating it.

RUN:
    pip install mlflow            # if you haven't
    python src/mlflow_tracking.py

THEN view the results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # open http://127.0.0.1:5000 in your browser
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the sibling module importable, then reuse its logic (DRY: one source of
# truth for features, split, params, training, and metrics).
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
import modeling as M  # noqa: E402

import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError:
        print("MLflow is not installed. Run:  pip install mlflow")
        return

    if not M.INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{M.INPUT_PATH} not found. Run features.py (and data_prep.py) first."
        )

    # MLflow 3.x deprecated the plain file store, so we use a local SQLite
    # backend (works across MLflow versions). .as_posix() keeps the URI valid on
    # Windows too. Metadata -> mlflow.db; artifacts -> ./mlartifacts (both
    # gitignored).
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("retail-sales-forecasting")

    # Prepare data exactly as modeling.py does.
    df = pd.read_parquet(M.INPUT_PATH)
    df = M.filter_open_positive(df)
    train, valid, cutoff = M.time_based_split(df)
    y_valid = valid[M.TARGET].to_numpy()

    with mlflow.start_run(run_name="xgboost-v1"):
        # --- params ---
        mlflow.log_param("model", "XGBoostRegressor")
        mlflow.log_params(M.XGB_PARAMS)
        mlflow.log_param("validation_days", M.VALIDATION_DAYS)
        mlflow.log_param("n_features", len(M.FEATURE_COLS))
        mlflow.log_param("n_train", len(train))
        mlflow.log_param("n_valid", len(valid))

        # --- baseline ---
        base_metrics = M.evaluate(y_valid, M.baseline_predict(valid))
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base_metrics.items()})

        # --- train + evaluate XGBoost ---
        print("Training XGBoost inside MLflow run ...")
        model = M.train_xgb(train)
        xgb_pred = np.clip(model.predict(valid[M.FEATURE_COLS]), 0, None)
        xgb_metrics = M.evaluate(y_valid, xgb_pred)
        mlflow.log_metrics({f"xgb_{k}": v for k, v in xgb_metrics.items()})

        # --- artifacts: log every chart we have, plus a fresh importance plot ---
        imp_path = FIG_DIR / "feature_importance.png"
        M.save_feature_importance(model, imp_path)
        if FIG_DIR.exists():
            for png in sorted(FIG_DIR.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="figures")

        # --- the model itself (handles both old and new MLflow signatures) ---
        try:
            mlflow.xgboost.log_model(model, name="model")
        except TypeError:
            mlflow.xgboost.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    print("\nLogged run:", run_id)
    print(f"baseline MAE={base_metrics['MAE']:.2f}  xgboost MAE={xgb_metrics['MAE']:.2f}")
    print("\nView it with:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("  then open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()