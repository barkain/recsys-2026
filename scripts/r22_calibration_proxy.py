#!/usr/bin/env python3
# ruff: noqa: T201
"""R22a: Calibration proxy — which local metrics predict blind nDCG transfer?

Loads the blind submission log, computes correlations, fits simple linear
models, and runs leave-one-out error analysis. Output: printed report +
saved JSON artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "data" / "r22_blind_submission_log.json"
OUT_PATH = REPO / "exp" / "eval" / "r22_calibration_proxy.json"


def load_log():
    with open(LOG_PATH) as f:
        return json.load(f)


def pearson(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), 0
    r = np.corrcoef(x, y)[0, 1]
    return float(r), int(mask.sum())


def linfit(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    residuals = y - pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {"slope": float(coeffs[0]), "intercept": float(coeffs[1]),
            "rmse": rmse, "n": int(len(x)),
            "residuals": {f"pt{i}": float(r) for i, r in enumerate(residuals)}}


def loo_error(x, y, names):
    """Leave-one-out linear fit error."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    idx = np.where(mask)[0]
    if len(idx) < 4:
        return None
    errors = {}
    for i in idx:
        train_mask = mask.copy()
        train_mask[i] = False
        xt, yt = x[train_mask], y[train_mask]
        if len(xt) < 2:
            continue
        coeffs = np.polyfit(xt, yt, 1)
        pred_i = np.polyval(coeffs, x[i])
        errors[names[i]] = {
            "actual": float(y[i]),
            "predicted": float(pred_i),
            "error": float(y[i] - pred_i),
            "abs_error": float(abs(y[i] - pred_i)),
        }
    mae = np.mean([v["abs_error"] for v in errors.values()])
    return {"loo_predictions": errors, "loo_mae": float(mae)}


def main():
    subs = load_log()
    names = [s["id"] for s in subs]
    blind = [s["actual_blind_ndcg"] for s in subs]

    local_metrics = {
        "local_cv5": [s.get("local_cv5") for s in subs],
        "pool_hit": [s.get("pool_hit") for s in subs],
        "pool_k": [s.get("pool_k") for s in subs],
        "n_features": [s.get("n_features") for s in subs],
        "n_sources": [s.get("n_sources") for s in subs],
    }

    # Derived metrics
    deltas_cv5 = [None]
    deltas_blind = [None]
    transfer_ratios = [None]
    for i in range(1, len(subs)):
        cv5_prev = subs[i - 1].get("local_cv5")
        cv5_curr = subs[i].get("local_cv5")
        b_prev = subs[i - 1]["actual_blind_ndcg"]
        b_curr = subs[i]["actual_blind_ndcg"]
        if cv5_prev is not None and cv5_curr is not None:
            d_cv5 = cv5_curr - cv5_prev
            d_blind = b_curr - b_prev
            deltas_cv5.append(d_cv5)
            deltas_blind.append(d_blind)
            transfer_ratios.append(d_blind / d_cv5 if abs(d_cv5) > 1e-6 else float("nan"))
        else:
            deltas_cv5.append(None)
            deltas_blind.append(None)
            transfer_ratios.append(None)

    local_metrics["delta_cv5"] = deltas_cv5
    local_metrics["delta_blind"] = deltas_blind
    local_metrics["transfer_ratio"] = transfer_ratios

    # ===== REPORT =====
    print("=" * 72)
    print("R22a CALIBRATION PROXY — BLIND SUBMISSION ANALYSIS")
    print("=" * 72)

    # Table 1: Submission history
    print("\n--- Submission History ---")
    print(f"{'ID':<16} {'Date':<12} {'Blind nDCG':>11} {'CV5':>8} {'Pool Hit':>9} "
          f"{'Pool K':>7} {'#Feat':>6} {'#Src':>5} {'Ranker':<15}")
    print("-" * 100)
    for s in subs:
        cv5 = f"{s['local_cv5']:.4f}" if s.get("local_cv5") else "?"
        ph = f"{s['pool_hit']:.4f}" if s.get("pool_hit") else "?"
        print(f"{s['id']:<16} {s['date']:<12} {s['actual_blind_ndcg']:>11.4f} {cv5:>8} "
              f"{ph:>9} {s.get('pool_k', '?'):>7} {s.get('n_features', '?'):>6} "
              f"{s.get('n_sources', '?'):>5} {s.get('ranker', '?'):<15}")

    # Table 2: Sequential deltas
    print("\n--- Sequential Deltas ---")
    print(f"{'Transition':<30} {'Δ CV5':>10} {'Δ Blind':>10} {'Transfer':>10}")
    print("-" * 65)
    for i in range(1, len(subs)):
        label = f"{subs[i-1]['id']} → {subs[i]['id']}"
        d_cv5 = deltas_cv5[i]
        d_blind = deltas_blind[i]
        tr = transfer_ratios[i]
        d_cv5_s = f"{d_cv5:+.4f}" if d_cv5 is not None else "?"
        d_blind_s = f"{d_blind:+.4f}"
        tr_s = f"{tr:+.2f}x" if tr is not None and np.isfinite(tr) else "?"
        print(f"{label:<30} {d_cv5_s:>10} {d_blind_s:>10} {tr_s:>10}")

    # Table 3: Correlations
    print("\n--- Correlations with Blind nDCG ---")
    print(f"{'Metric':<20} {'Pearson r':>10} {'N':>5}")
    print("-" * 40)
    corr_results = {}
    for metric_name, values in local_metrics.items():
        r, n = pearson(values, blind)
        corr_results[metric_name] = {"pearson_r": r, "n": n}
        r_s = f"{r:.4f}" if np.isfinite(r) else "n/a"
        print(f"{metric_name:<20} {r_s:>10} {n:>5}")

    # Table 4: Linear fits
    print("\n--- Linear Fits: local_metric → blind_ndcg ---")
    fit_results = {}
    for metric_name in ["local_cv5", "pool_hit", "pool_k", "n_sources"]:
        values = local_metrics[metric_name]
        fit = linfit(values, blind)
        if fit:
            fit_results[metric_name] = fit
            print(f"\n  {metric_name}:")
            print(f"    blind_ndcg = {fit['slope']:.4f} * {metric_name} + {fit['intercept']:.4f}")
            print(f"    RMSE = {fit['rmse']:.4f}  (n={fit['n']})")

    # Table 5: LOO predictions for key metrics
    print("\n--- Leave-One-Out Predictions ---")
    loo_results = {}
    for metric_name in ["local_cv5", "pool_hit"]:
        values = local_metrics[metric_name]
        loo = loo_error(values, blind, names)
        if loo:
            loo_results[metric_name] = loo
            print(f"\n  {metric_name} (LOO MAE = {loo['loo_mae']:.4f}):")
            print(f"    {'Submission':<16} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
            print(f"    {'-'*45}")
            for sub_name, vals in loo["loo_predictions"].items():
                print(f"    {sub_name:<16} {vals['actual']:>8.4f} {vals['predicted']:>10.4f} "
                      f"{vals['error']:>+8.4f}")

    # Key findings
    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)

    # 1. Which metrics correlate?
    sorted_corrs = sorted(corr_results.items(),
                          key=lambda x: abs(x[1]["pearson_r"]) if np.isfinite(x[1]["pearson_r"]) else 0,
                          reverse=True)
    print("\n1. WHICH LOCAL METRICS CORRELATE WITH BLIND nDCG TRANSFER?")
    for name, vals in sorted_corrs[:5]:
        if vals["n"] >= 3 and np.isfinite(vals["pearson_r"]):
            print(f"   {name}: r={vals['pearson_r']:.3f} (n={vals['n']})")

    # 2. Local lift without blind transfer
    print("\n2. EXPERIMENTS WITH LOCAL LIFT BUT POOR BLIND TRANSFER:")
    for i in range(1, len(subs)):
        d_cv5 = deltas_cv5[i]
        d_blind = deltas_blind[i]
        if d_cv5 is not None and d_cv5 > 0.005 and d_blind is not None and d_blind <= 0.005:
            print(f"   {subs[i]['id']}: Δcv5={d_cv5:+.4f}, Δblind={d_blind:+.4f}  ← {subs[i].get('notes','')[:80]}")

    # 3. R21 positioning
    print("\n3. WHERE R21 SITS RELATIVE TO PRIOR TRANSFER PATTERNS:")
    r21 = next(s for s in subs if s["id"] == "r21")
    if "local_cv5" in fit_results:
        fit = fit_results["local_cv5"]
        predicted = fit["slope"] * r21["local_cv5"] + fit["intercept"]
        residual = r21["actual_blind_ndcg"] - predicted
        print(f"   R21 local_cv5={r21['local_cv5']:.4f} → linear prediction: {predicted:.4f}")
        print(f"   R21 actual blind={r21['actual_blind_ndcg']:.4f} → residual: {residual:+.4f}")
        if residual > 0.01:
            print("   R21 OUTPERFORMS the linear trend — supervised retrieval transfers better than expected.")
        elif residual < -0.01:
            print("   R21 UNDERPERFORMS the linear trend — may be overfitting locally.")
        else:
            print("   R21 is on-trend — linear proxy is reasonable for this point.")

    if "pool_hit" in fit_results:
        fit = fit_results["pool_hit"]
        predicted = fit["slope"] * r21["pool_hit"] + fit["intercept"]
        residual = r21["actual_blind_ndcg"] - predicted
        print(f"   R21 pool_hit={r21['pool_hit']:.4f} → linear prediction: {predicted:.4f}")
        print(f"   Residual: {residual:+.4f}")

    # 4. Proxy recommendation
    print("\n4. PROXY RECOMMENDATION FOR R22b GATE:")
    best_metric = sorted_corrs[0][0] if sorted_corrs else "local_cv5"
    best_r = sorted_corrs[0][1]["pearson_r"] if sorted_corrs else 0
    print(f"   Best single predictor: {best_metric} (r={best_r:.3f})")
    if best_metric in loo_results:
        print(f"   LOO MAE: {loo_results[best_metric]['loo_mae']:.4f}")
    print(f"   With 8 data points, use linear fit as directional gate, not precise prediction.")
    print(f"   R15 lesson: local lift without pool coverage improvement doesn't transfer.")

    # Save artifact
    artifact = {
        "submissions": [{
            "id": s["id"], "name": s["name"], "date": s["date"],
            "blind_ndcg": s["actual_blind_ndcg"],
            "local_cv5": s.get("local_cv5"),
            "pool_hit": s.get("pool_hit"),
            "pool_k": s.get("pool_k"),
            "n_sources": s.get("n_sources"),
            "n_features": s.get("n_features"),
        } for s in subs],
        "correlations": corr_results,
        "linear_fits": fit_results,
        "loo_analysis": loo_results,
        "sequential_deltas": [{
            "from": subs[i-1]["id"], "to": subs[i]["id"],
            "delta_cv5": deltas_cv5[i], "delta_blind": deltas_blind[i],
            "transfer_ratio": transfer_ratios[i] if transfer_ratios[i] is not None and np.isfinite(transfer_ratios[i]) else None,
        } for i in range(1, len(subs))],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
