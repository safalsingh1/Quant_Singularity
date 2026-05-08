"""
eval_suite.py — Walk-Forward Evaluation Suite
COMMIT THIS BEFORE THE FIRST KAGGLE TRAINING RUN.

Defines ALL metrics and thresholds pre-training, as required by the brief.
Evaluation: Days 31–60, rolling 5-day windows. Walk-forward only — no k-fold.
"""
import json, logging, math, numpy as np, pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("eval_suite")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DATA_DIR   = Path(__file__).parent / "slm_intern_data"
MARKET_FILE = DATA_DIR / "market_states.parquet"
RESULTS_DIR = Path(__file__).parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# PRE-TRAINING THRESHOLD COMMITMENTS (written before training)
# ═══════════════════════════════════════════════════════════
THRESHOLDS = {
    # Minimum directional accuracy over the full eval window (days 31–60)
    "min_directional_accuracy":       0.45,   # Better than random (3-class = 0.33)
    # Per-window floor: any 5-day window below this triggers review
    "min_per_window_accuracy":        0.38,
    # Schema pass rate: every single call must produce valid JSON
    "min_schema_pass_rate":           0.95,   # <95% is a hard fail
    "target_schema_pass_rate":        1.00,   # We aim for 100%
    # Conviction: most signals should be >0.4 (otherwise orchestrator would always downgrade)
    "min_mean_conviction_passing":    0.42,
    # Orchestrator suppression by rule
    "max_adx_suppression_rate":       0.25,   # ADX<20 suppresses; >25% means mostly ranging market
    "max_parse_failure_rate":         0.02,   # <2% parse failures
    "max_conviction_downgrade_rate":  0.45,   # <45% downgraded (signals should have conviction)
    # Regime slicing thresholds
    "high_vix_accuracy_floor":        0.38,   # Lower bar in stress regime
    "low_vix_accuracy_floor":         0.45,   # Higher bar in calm regime
    # Calibration: ECE (Expected Calibration Error) of conviction vs accuracy
    "max_conviction_ece":             0.20,   # Below 0.20 is acceptable
}


def load_eval_data() -> pd.DataFrame:
    df = pd.read_parquet(MARKET_FILE)
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Days 31–60 = second half (eval window). We identify them by unique trading days.
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    unique_days = sorted(df["date"].unique())
    eval_days   = set(unique_days[30:60])  # 0-indexed: day index 30..59
    eval_df     = df[df["date"].isin(eval_days)].copy()
    logger.info("Eval set: %d rows across %d trading days", len(eval_df), len(eval_days))
    return eval_df


def directional_accuracy(labels: List[str], preds: List[str]) -> float:
    if not labels:
        return float("nan")
    return sum(l==p for l,p in zip(labels,preds)) / len(labels)


def schema_pass_rate(signals: List[dict]) -> float:
    valid = 0
    for s in signals:
        try:
            assert s.get("direction") in {"CE","PE","NEUTRAL"}
            assert isinstance(s.get("conviction"), (int,float))
            assert 0.0 <= float(s["conviction"]) <= 1.0
            assert s.get("horizon") in {"intraday","next_session"}
            assert s.get("signal_id")
            assert s.get("generated_at")
            valid += 1
        except (AssertionError, TypeError, KeyError):
            pass
    return valid / len(signals) if signals else float("nan")


def conviction_ece(convictions: List[float], accuracies: List[float],
                   n_bins: int = 5) -> float:
    """Expected Calibration Error across equal-width conviction bins."""
    if not convictions:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(convictions)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = [(lo <= c < hi) for c in convictions]
        if not any(mask):
            continue
        b_conv = [c for c,m in zip(convictions,mask) if m]
        b_acc  = [a for a,m in zip(accuracies, mask) if m]
        ece += (len(b_conv)/n) * abs(np.mean(b_conv) - np.mean(b_acc))
    return ece


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def evaluate_orchestrator_output(eval_df: pd.DataFrame, orch_signals: List[dict],
                                 use_rag: bool = False) -> dict:
    """
    Full eval: given orchestrator outputs for each eval row, compute all metrics.
    eval_df must have a 'label' column (ground-truth direction).
    """
    assert len(eval_df) == len(orch_signals), \
        f"Mismatch: eval_df={len(eval_df)}, signals={len(orch_signals)}"

    labels  = list(eval_df["label"])
    dates   = list(eval_df["date"])
    vix_col = list(eval_df["vix_india"])
    adx_col = list(eval_df["adx_14"])

    # Split into 5-day rolling windows
    unique_days = sorted(set(dates))
    windows = []
    for i in range(0, len(unique_days), 5):
        w_days = set(unique_days[i:i+5])
        idxs   = [j for j,d in enumerate(dates) if d in w_days]
        windows.append({"days": sorted(w_days), "idxs": idxs})

    # Per-signal metrics
    final_dirs   = [s["direction"] for s in orch_signals]
    convictions  = [float(s.get("conviction", 0)) for s in orch_signals]
    actions      = [s.get("orchestrator_action","?") for s in orch_signals]
    schema_ok    = schema_pass_rate(orch_signals)

    # Global accuracy (non-suppressed only)
    active_mask  = [a not in ("SUPPRESSED_ADX","SUPPRESSED_PARSE") for a in actions]
    active_labels= [l for l,m in zip(labels,active_mask) if m]
    active_preds = [p for p,m in zip(final_dirs,active_mask) if m]
    global_acc   = directional_accuracy(active_labels, active_preds)
    n_active     = len(active_labels)
    global_ci    = wilson_ci(sum(l==p for l,p in zip(active_labels,active_preds)), n_active)

    # Orchestrator rates
    n_total = len(orch_signals)
    n_supp_adx  = sum(1 for a in actions if a=="SUPPRESSED_ADX")
    n_supp_parse= sum(1 for a in actions if a=="SUPPRESSED_PARSE")
    n_downgrade  = sum(1 for a in actions if a=="DOWNGRADED_CONVICTION")
    n_pass       = sum(1 for a in actions if a=="PASS_THROUGH")

    # VIX regime slicing (high = top quartile of eval set vix)
    vix_arr  = np.array(vix_col)
    vix_q75  = np.percentile(vix_arr, 75)
    high_vix = vix_arr >= vix_q75
    low_vix  = ~high_vix

    hv_labels= [l for l,m in zip(labels,high_vix) if m and active_mask[labels.index(l) if l in labels else 0]]
    # Simpler iteration:
    hv_pairs = [(labels[i], final_dirs[i]) for i in range(n_total) if high_vix[i] and active_mask[i]]
    lv_pairs = [(labels[i], final_dirs[i]) for i in range(n_total) if low_vix[i] and active_mask[i]]
    hv_acc   = directional_accuracy([x[0] for x in hv_pairs], [x[1] for x in hv_pairs])
    lv_acc   = directional_accuracy([x[0] for x in lv_pairs], [x[1] for x in lv_pairs])

    # Conviction ECE
    passing_conv = [convictions[i] for i in range(n_total) if active_mask[i]]
    passing_acc_binary = [1 if labels[i]==final_dirs[i] else 0
                          for i in range(n_total) if active_mask[i]]
    ece = conviction_ece(passing_conv, [float(x) for x in passing_acc_binary])

    # Per-window accuracy
    window_results = []
    for w in windows:
        idxs = w["idxs"]
        w_active = [(labels[i], final_dirs[i]) for i in idxs if active_mask[i]]
        w_acc = directional_accuracy([x[0] for x in w_active], [x[1] for x in w_active])
        w_ci  = wilson_ci(sum(x[0]==x[1] for x in w_active), len(w_active))
        window_results.append({
            "window_start": str(w["days"][0]),
            "window_end":   str(w["days"][-1]),
            "n_total":      len(idxs),
            "n_active":     len(w_active),
            "accuracy":     round(w_acc, 4) if not math.isnan(w_acc) else None,
            "ci_low":       round(w_ci[0], 4),
            "ci_high":      round(w_ci[1], 4),
            "meets_threshold": (w_acc >= THRESHOLDS["min_per_window_accuracy"])
                               if not math.isnan(w_acc) else False,
        })

    results = {
        "use_rag":              use_rag,
        "n_eval_rows":          n_total,
        "schema_pass_rate":     round(schema_ok, 4),
        "schema_ok":            schema_ok >= THRESHOLDS["min_schema_pass_rate"],
        "global_accuracy":      round(global_acc, 4) if not math.isnan(global_acc) else None,
        "global_acc_ci":        [round(x,4) for x in global_ci],
        "global_acc_ok":        (global_acc >= THRESHOLDS["min_directional_accuracy"])
                                if not math.isnan(global_acc) else False,
        "n_active":             n_active,
        "n_suppressed_adx":     n_supp_adx,
        "n_suppressed_parse":   n_supp_parse,
        "n_downgraded":         n_downgrade,
        "n_pass_through":       n_pass,
        "adx_suppression_rate":  round(n_supp_adx/n_total, 4) if n_total else 0,
        "parse_failure_rate":    round(n_supp_parse/n_total, 4) if n_total else 0,
        "conviction_downgrade_rate": round(n_downgrade/n_total, 4) if n_total else 0,
        "high_vix_accuracy":    round(hv_acc, 4) if not math.isnan(hv_acc) else None,
        "low_vix_accuracy":     round(lv_acc, 4) if not math.isnan(lv_acc) else None,
        "high_vix_threshold":   round(float(vix_q75), 2),
        "conviction_ece":       round(ece, 4) if not math.isnan(ece) else None,
        "mean_conviction_passing": round(float(np.mean(passing_conv)),4) if passing_conv else None,
        "window_results":       window_results,
        "thresholds_used":      THRESHOLDS,
    }

    # Save
    tag = "rag" if use_rag else "no_rag"
    out_file = RESULTS_DIR / f"eval_{tag}.json"
    out_file.write_text(json.dumps(results, indent=2), "utf-8")
    logger.info("Eval results saved to %s", out_file)

    _print_summary(results)
    return results


def _print_summary(r: dict):
    tag = "WITH RAG" if r["use_rag"] else "WITHOUT RAG"
    lines = [
        f"\n{'='*60}",
        f"EVAL SUMMARY [{tag}]",
        f"{'='*60}",
        f"Schema pass rate   : {r['schema_pass_rate']:.3f}  ({'OK' if r['schema_ok'] else 'FAIL'})",
        f"Global accuracy    : {r['global_accuracy']}  95%CI={r['global_acc_ci']}  ({'OK' if r['global_acc_ok'] else 'FAIL'})",
        f"High-VIX accuracy  : {r['high_vix_accuracy']}  (VIX >= {r['high_vix_threshold']})",
        f"Low-VIX  accuracy  : {r['low_vix_accuracy']}",
        f"ADX suppress rate  : {r['adx_suppression_rate']:.3f}",
        f"Parse fail rate    : {r['parse_failure_rate']:.3f}",
        f"Conv downgrade rate: {r['conviction_downgrade_rate']:.3f}",
        f"Conviction ECE     : {r['conviction_ece']}",
        f"Mean conviction    : {r['mean_conviction_passing']}",
        f"\n5-Day Window Breakdown:",
    ]
    for w in r["window_results"]:
        ok = "OK" if w["meets_threshold"] else "!!"
        lines.append(f"  {w['window_start']}–{w['window_end']}: acc={w['accuracy']} "
                     f"CI=[{w['ci_low']},{w['ci_high']}] n={w['n_active']} [{ok}]")
    lines.append("="*60)
    print("\n".join(lines))


def run_mock_eval(use_rag: bool = False) -> dict:
    """
    Run eval using a mock pod (random signals + fallbacks) for pipeline testing.
    This lets us validate the eval machinery before training completes.
    """
    import random
    eval_df = load_eval_data()
    mock_signals = []
    from orchestrator import Orchestrator, OrchestratorDecision
    orch = Orchestrator(pod=None, use_rag=False)

    class MockPod:
        def predict(self, ms, **kw):
            import uuid
            from datetime import datetime, timezone
            direction = random.choice(["CE","PE","NEUTRAL"])
            conviction = round(random.uniform(0.25, 0.75), 2)
            return {
                "direction": direction,
                "conviction": conviction,
                "horizon": random.choice(["intraday","next_session"]),
                "signal_id": str(uuid.uuid4()),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    orch._pod = MockPod()
    for _, row in eval_df.iterrows():
        ms = row.to_dict()
        ms.pop("date", None)
        ms.pop("label", None)
        sig = orch.process(ms)
        mock_signals.append(sig)

    return evaluate_orchestrator_output(eval_df, mock_signals, use_rag=use_rag)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "mock"
    if mode == "mock":
        print("Running MOCK eval (random pod) to validate eval pipeline...")
        run_mock_eval(use_rag=False)
        run_mock_eval(use_rag=True)
    else:
        print("Run eval_suite via run_eval.py for live pod evaluation.")
