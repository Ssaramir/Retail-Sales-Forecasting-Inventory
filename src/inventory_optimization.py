"""
inventory_optimization.py
-------------------------
Turn forecasts into OPTIMIZED daily order quantities with a linear program
(PuLP) that minimizes holding + stockout cost subject to lead-time, storage, and
SAFETY-STOCK (service-level) constraints. The LP is run in a ROLLING-HORIZON loop
(re-solve each day with the real inventory state, commit only that day's order —
Model Predictive Control style) so it can react to forecast error, then compared
honestly against a simple order-up-to policy on ACTUAL demand.

Why this design (the honest story):
  A deterministic, open-loop LP that trusts a point forecast holds no buffer and
  gets punished when actual demand exceeds the forecast. Two standard fixes:
    1. a safety-stock floor  I_t >= safety  (sized from recent forecast error),
       so the plan carries a buffer for uncertainty;
    2. rolling re-optimization, so the plan adapts each day to what really sold.

THE LP (over a horizon, per store):
  vars >= 0:  o_t order, I_t end inventory, m_t demand met, s_t shortage
  min   sum_t ( h*I_t + p*s_t )                      # holding + stockout
  s.t.  I_t = I_{t-1} + arrivals_t - m_t              # balance
        arrivals_t = committed pipeline + o_{t-L}     # LEAD TIME
        m_t <= d_t ;  s_t = d_t - m_t                 # demand / shortage
        I_t <= C                                      # STORAGE CAPACITY
        I_t >= safety   (for t >= lead time)          # SAFETY STOCK

INPUT:  outputs/predictions/validation_predictions.parquet
OUTPUTS:
  outputs/predictions/optimization_plan.parquet
  outputs/metrics/optimization_summary.json
  outputs/figures/opt_cost_comparison.png
  outputs/figures/opt_store_plan.png

RUN:
    pip install pulp
    python src/inventory_optimization.py
    # quicker trial: OPT_N_STORES=10 python src/inventory_optimization.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
PRED_PATH = OUT / "predictions" / "validation_predictions.parquet"

# --- Config (illustrative cost/constraint assumptions; override via env) ----
N_STORES = int(os.environ.get("OPT_N_STORES", 30))
LEAD_TIME = int(os.environ.get("OPT_LEAD_TIME", 2))         # days until an order arrives
HOLDING_COST = float(os.environ.get("OPT_HOLDING", 0.05))   # cost per unit held per day
STOCKOUT_COST = float(os.environ.get("OPT_STOCKOUT", 5.0))  # cost per unit short (>> holding)
STORAGE_CAP_MULT = float(os.environ.get("OPT_CAP_MULT", 3.0))  # capacity = mult x mean demand
SERVICE_Z = 1.645   # ~95% buffer multiplier for safety stock & the baseline policy


def optimize_store(demand, I0, L, C, h, p, safety=0.0, fixed_arrivals=None):
    """Solve the inventory LP over `demand` (a forecast horizon). `fixed_arrivals`
    are units already in transit that will arrive on each day. The safety floor is
    applied from day L onward (earlier days can't be lifted within the lead time)."""
    T = len(demand)
    if fixed_arrivals is None:
        fixed_arrivals = np.zeros(T)
    prob = pulp.LpProblem("store_inventory", pulp.LpMinimize)
    o = [pulp.LpVariable(f"o_{t}", lowBound=0) for t in range(T)]
    I = [pulp.LpVariable(f"I_{t}", lowBound=0) for t in range(T)]
    m = [pulp.LpVariable(f"m_{t}", lowBound=0) for t in range(T)]
    s = [pulp.LpVariable(f"s_{t}", lowBound=0) for t in range(T)]

    for t in range(T):
        new_arr = o[t - L] if t - L >= 0 else 0
        arrivals = fixed_arrivals[t] + new_arr
        prev = I[t - 1] if t >= 1 else I0
        prob += I[t] == prev + arrivals - m[t]      # balance
        prob += m[t] <= demand[t]                   # sell at most demand
        prob += s[t] == demand[t] - m[t]            # shortage
        prob += I[t] <= C                           # capacity
        if t >= L:
            prob += I[t] >= safety                  # safety stock (service level)

    prob += pulp.lpSum(h * I[t] + p * s[t] for t in range(T))
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    val = lambda v: float(v.value() or 0.0)
    return {
        "orders": np.array([val(x) for x in o]),
        "inventory": np.array([val(x) for x in I]),
        "shortage": np.array([val(x) for x in s]),
        "cost": float(pulp.value(prob.objective)),
        "status": pulp.LpStatus[prob.status],
    }


def realized_cost_of_orders(orders, demand, I0, L, h, p):
    """Play a fixed order schedule forward against ACTUAL demand."""
    T = len(demand)
    arrivals = np.zeros(T + L + 1)
    onhand, cost = I0, 0.0
    inv, short = np.zeros(T), np.zeros(T)
    for t in range(T):
        onhand += arrivals[t]
        met = min(demand[t], onhand)
        sh = demand[t] - met
        onhand -= met
        cost += h * onhand + p * sh
        inv[t], short[t] = onhand, sh
        arrivals[t + L] += orders[t]
    return cost, inv, short


def rolling_horizon(forecast, actual, I0, L, C, h, p, safety):
    """Model-Predictive-Control loop: each day re-solve the LP over the remaining
    forecast with the real current inventory + committed pipeline, commit only the
    first order, then advance one day on ACTUAL demand."""
    T = len(forecast)
    pipeline = np.zeros(T + L + 1)        # committed arrivals by day
    onhand, cost = I0, 0.0
    orders, inv, short = np.zeros(T), np.zeros(T), np.zeros(T)
    for k in range(T):
        rem = forecast[k:]
        fixed = pipeline[k:k + len(rem)].copy()
        res = optimize_store(rem, onhand, L, C, h, p, safety, fixed_arrivals=fixed)
        o0 = res["orders"][0]
        orders[k] = o0
        onhand += pipeline[k]             # receive today's committed arrivals
        met = min(actual[k], onhand)
        sh = actual[k] - met
        onhand -= met
        cost += h * onhand + p * sh
        inv[k], short[k] = onhand, sh
        pipeline[k + L] += o0             # this order arrives after lead time
    return orders, inv, short, cost


def simulate_order_up_to(demand, I0, L, S, h, p):
    """Closed-loop baseline: each day order up to level S; realized on actuals."""
    T = len(demand)
    arrivals = np.zeros(T + L + 1)
    onhand, cost = I0, 0.0
    for t in range(T):
        onhand += arrivals[t]
        met = min(demand[t], onhand)
        sh = demand[t] - met
        onhand -= met
        cost += h * onhand + p * sh
        outstanding = arrivals[t + 1: t + L + 1].sum()
        arrivals[t + L] += max(0.0, S - (onhand + outstanding))
    return cost


def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"{PRED_PATH} not found. Run modeling.py first.")

    preds = pd.read_parquet(PRED_PATH)
    stores = sorted(preds["Store"].unique())[:N_STORES]
    print(f"Optimizing {len(stores)} stores | lead time {LEAD_TIME}d | "
          f"holding {HOLDING_COST}/unit/day | stockout {STOCKOUT_COST}/unit | "
          f"rolling-horizon + safety stock\n")

    plan_rows = []
    cost_openloop, cost_rolling, cost_policy, infeasible = 0.0, 0.0, 0.0, 0
    example = None

    for store in stores:
        g = preds[preds["Store"] == store].sort_values("Date")
        forecast = g["y_pred_xgb"].to_numpy()
        actual = g["Sales"].to_numpy()
        if len(forecast) < LEAD_TIME + 2:
            continue
        mean_d, std_d = float(forecast.mean()), float(forecast.std())
        err_std = float((actual - forecast).std())     # forecast-error spread
        safety = SERVICE_Z * err_std                    # safety stock
        C = STORAGE_CAP_MULT * mean_d
        I0 = mean_d

        # (a) open-loop LP with safety, realized on actuals (for context)
        res = optimize_store(forecast, I0, LEAD_TIME, C, HOLDING_COST,
                             STOCKOUT_COST, safety)
        if res["status"] != "Optimal":
            infeasible += 1
            continue
        ol_cost, _, _ = realized_cost_of_orders(res["orders"], actual, I0,
                                                LEAD_TIME, HOLDING_COST, STOCKOUT_COST)
        cost_openloop += ol_cost

        # (b) rolling-horizon LP (the real deliverable)
        orders, inv, short, rl_cost = rolling_horizon(
            forecast, actual, I0, LEAD_TIME, C, HOLDING_COST, STOCKOUT_COST, safety)
        cost_rolling += rl_cost

        # (c) simple order-up-to baseline
        S = mean_d * (LEAD_TIME + 1) + SERVICE_Z * std_d
        cost_policy += simulate_order_up_to(actual, I0, LEAD_TIME, S,
                                            HOLDING_COST, STOCKOUT_COST)

        for date, d, o, iv, sh in zip(g["Date"], forecast, orders, inv, short):
            plan_rows.append({"Store": store, "Date": date,
                              "forecast_demand": round(float(d), 1),
                              "order_qty": round(float(o), 1),
                              "end_inventory": round(float(iv), 1),
                              "shortage": round(float(sh), 1)})
        if example is None:
            example = (store, g["Date"].to_numpy(), forecast, orders, inv)

    plan = pd.DataFrame(plan_rows)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    plan.to_parquet(OUT / "predictions" / "optimization_plan.parquet", index=False)

    pct = (cost_policy - cost_rolling) / cost_policy * 100 if cost_policy else 0.0
    summary = {
        "n_stores": len(stores), "lead_time_days": LEAD_TIME,
        "holding_cost_per_unit_day": HOLDING_COST, "stockout_cost_per_unit": STOCKOUT_COST,
        "storage_cap_multiple": STORAGE_CAP_MULT, "service_z": SERVICE_Z,
        "order_up_to_cost": round(cost_policy, 1),
        "lp_open_loop_cost": round(cost_openloop, 1),
        "lp_rolling_horizon_cost": round(cost_rolling, 1),
        "rolling_lp_cost_reduction_pct": round(pct, 1),
        "infeasible_stores": infeasible,
        "note": "Realized cost on ACTUAL demand; costs are illustrative assumptions.",
    }
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "metrics" / "optimization_summary.json", "w"), indent=2)

    print("=" * 64)
    print("REALIZED COST ON ACTUAL DEMAND (lower is better)")
    print("=" * 64)
    print(f"  Simple order-up-to policy      : {cost_policy:>12,.0f}")
    print(f"  LP open-loop (+ safety)        : {cost_openloop:>12,.0f}")
    print(f"  LP rolling-horizon (+ safety)  : {cost_rolling:>12,.0f}")
    print(f"  Rolling LP vs order-up-to      : {pct:>11.1f}%  cost reduction")
    print(f"  Infeasible stores              : {infeasible}")

    # --- charts ---
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(["Order-up-to", "LP open-loop", "LP rolling"],
           [cost_policy, cost_openloop, cost_rolling],
           color=["#94A3B8", "#7FB3B8", "#028090"])
    ax.set_title("Realized cost on actual demand (lower is better)")
    ax.set_ylabel("Total holding + stockout cost")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "opt_cost_comparison.png", dpi=120); plt.close(fig)

    if example is not None:
        store, dates, demand, orders, inv = example
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(dates, demand, color="#1E293B", lw=2, label="Forecast demand")
        ax.bar(dates, orders, color="#028090", alpha=0.6, label="Order qty")
        ax.plot(dates, inv, color="#E08A1E", lw=2, ls="--", label="End inventory")
        ax.set_title(f"Store {store}: rolling-horizon LP order plan")
        ax.set_ylabel("Units"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "figures" / "opt_store_plan.png", dpi=120); plt.close(fig)

    print(f"\nSaved optimization_plan.parquet, optimization_summary.json, and 2 figures -> outputs/")
    print("Done.")


if __name__ == "__main__":
    main()