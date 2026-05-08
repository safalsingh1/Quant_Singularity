"""
run_eval.py — Run the full eval suite against a trained pod (with and without RAG).
Usage:
    python run_eval.py              # uses local adapter
    python run_eval.py --no-rag    # only runs without RAG condition
    python run_eval.py --mock      # mock pod (pipeline test)
    python run_eval.py --device cuda  # use GPU
"""
import argparse, json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("run_eval")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock",      action="store_true", help="Use mock random pod")
    p.add_argument("--no-rag",    action="store_true", help="Only run no-RAG condition")
    p.add_argument("--rag-only",  action="store_true", help="Only run RAG condition")
    p.add_argument("--device",    default="cpu",        help="cpu|cuda")
    p.add_argument("--adapter",   default=None,         help="Path to LoRA adapter")
    p.add_argument("--no-4bit",   action="store_true",  help="Disable 4-bit quantisation")
    return p.parse_args()


def main():
    args = parse_args()

    from eval_suite import load_eval_data, evaluate_orchestrator_output, run_mock_eval
    from orchestrator import Orchestrator

    if args.mock:
        logger.info("Running MOCK eval (pipeline validation)...")
        run_mock_eval(use_rag=False)
        if not args.no_rag:
            run_mock_eval(use_rag=True)
        return

    from pod import SignalPod
    pod = SignalPod(
        adapter_path=args.adapter,
        use_4bit=not args.no_4bit,
        device=args.device,
    )

    eval_df = load_eval_data()

    conditions = []
    if not args.rag_only:
        conditions.append(False)   # no-RAG
    if not args.no_rag:
        conditions.append(True)    # RAG

    all_results = {}
    for use_rag in conditions:
        logger.info("Running eval: use_rag=%s", use_rag)
        orch = Orchestrator(pod=pod, use_rag=use_rag)
        signals = []
        for _, row in eval_df.iterrows():
            ms = {k: row[k] for k in row.index if k not in ("date","label")}
            sig = orch.process(ms)
            signals.append(sig)
        result = evaluate_orchestrator_output(eval_df, signals, use_rag=use_rag)
        all_results["rag" if use_rag else "no_rag"] = result

    # Compare RAG vs no-RAG if both present
    if "rag" in all_results and "no_rag" in all_results:
        r_rag   = all_results["rag"]
        r_norag = all_results["no_rag"]
        delta_acc  = (r_rag["global_accuracy"] or 0) - (r_norag["global_accuracy"] or 0)
        delta_conv = (r_rag["mean_conviction_passing"] or 0) - (r_norag["mean_conviction_passing"] or 0)
        print("\n" + "="*60)
        print("RAG ABLATION COMPARISON")
        print("="*60)
        print(f"Accuracy delta (RAG - no-RAG): {delta_acc:+.4f}")
        print(f"Conviction delta (RAG - no-RAG): {delta_conv:+.4f}")
        interp = "RAG HELPS" if delta_acc > 0.01 else ("RAG HURTS" if delta_acc < -0.01 else "RAG NEUTRAL")
        print(f"Interpretation: {interp}")
        rag_summary = {
            "delta_accuracy": round(delta_acc, 4),
            "delta_conviction": round(delta_conv, 4),
            "interpretation": interp,
        }
        from eval_suite import RESULTS_DIR
        (RESULTS_DIR / "rag_ablation.json").write_text(json.dumps(rag_summary, indent=2), "utf-8")

    logger.info("Eval complete.")


if __name__ == "__main__":
    main()
