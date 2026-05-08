"""
orchestrator.py — Quant Singularity Signal Orchestrator
Wraps the SignalPod with three sequential suppression/downgrade rules.
Logs every decision with reason code and triggering values.
Downstream pipeline reads ONLY orchestrator output, never raw pod signal.
"""
import json, logging, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

LOG_FILE = Path(__file__).parent / "orchestrator_decisions.jsonl"

# Thresholds (fixed per brief)
ADX_SUPPRESSION_THRESHOLD = 20.0
CONVICTION_DOWNGRADE_THRESHOLD = 0.40


class OrchestratorDecision:
    PASS_THROUGH       = "PASS_THROUGH"
    SUPPRESSED_ADX     = "SUPPRESSED_ADX"
    SUPPRESSED_PARSE   = "SUPPRESSED_PARSE"
    DOWNGRADED_CONVICTION = "DOWNGRADED_CONVICTION"


def _neutral_signal(reason_code: str, values: dict, original: Optional[dict] = None) -> dict:
    return {
        "direction":   "NEUTRAL",
        "conviction":  0.0,
        "horizon":     (original or {}).get("horizon", "intraday"),
        "signal_id":   str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "orchestrator_action": reason_code,
        "orchestrator_values": values,
    }


def _log_decision(market_state: dict, pod_signal: Optional[dict], final_signal: dict,
                  action: str, values: dict):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "values": values,
        "pod_direction":   (pod_signal or {}).get("direction"),
        "pod_conviction":  (pod_signal or {}).get("conviction"),
        "final_direction": final_signal["direction"],
        "final_conviction": final_signal["conviction"],
        "market_state_summary": {
            k: market_state.get(k)
            for k in ["adx_14","vix_india","pcr","dte_nearest","moneyness_band"]
        }
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("ORCH action=%s adx=%.2f conviction=%s final=%s",
                action,
                market_state.get("adx_14", 0),
                (pod_signal or {}).get("conviction"),
                final_signal["direction"])


class Orchestrator:
    """
    Three rules applied in sequence:
      1. If ADX < 20 → suppress entirely, return NEUTRAL. Do NOT call pod.
      2. If pod output fails to parse → return NEUTRAL, log raw output.
      3. If conviction < 0.40 → downgrade direction to NEUTRAL (keep conviction).
    Downstream pipeline only sees Orchestrator output.
    """

    def __init__(self, pod=None, use_rag: bool = False):
        self._pod = pod
        self._use_rag = use_rag
        self._retrieve = None
        if use_rag:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent / "slm_intern_data"))
                from retrieve import retrieve
                self._retrieve = retrieve
            except ImportError:
                logger.warning("RAG retrieve not available")

    def _get_pod(self):
        if self._pod is None:
            from pod import get_pod
            self._pod = get_pod()
        return self._pod

    def process(self, market_state: dict) -> dict:
        """
        Main entry point. Given a market_state dict, return an orchestrated signal dict.
        """
        adx = float(market_state.get("adx_14", 0))

        # ─── Rule 1: ADX suppression ───────────────────────────────────────
        if adx < ADX_SUPPRESSION_THRESHOLD:
            values = {"adx_14": adx, "threshold": ADX_SUPPRESSION_THRESHOLD}
            final = _neutral_signal(OrchestratorDecision.SUPPRESSED_ADX, values)
            _log_decision(market_state, None, final,
                          OrchestratorDecision.SUPPRESSED_ADX, values)
            return final

        # ─── Retrieve RAG context (optional) ──────────────────────────────
        retrieved = None
        if self._use_rag and self._retrieve:
            try:
                retrieved = self._retrieve(market_state, k=3)
            except Exception as e:
                logger.warning("RAG retrieve failed: %s", e)

        # ─── Call pod ─────────────────────────────────────────────────────
        pod = self._get_pod()
        pod_signal = pod.predict(market_state, retrieved_episodes=retrieved)

        # ─── Rule 2: Parse failure suppression ────────────────────────────
        if pod_signal.get("_fallback") or pod_signal.get("direction") not in {"CE","PE","NEUTRAL"}:
            raw = pod_signal.get("_fallback_reason","unknown")
            values = {"reason": raw}
            final = _neutral_signal(OrchestratorDecision.SUPPRESSED_PARSE, values, pod_signal)
            _log_decision(market_state, pod_signal, final,
                          OrchestratorDecision.SUPPRESSED_PARSE, values)
            return final

        # ─── Rule 3: Conviction downgrade ─────────────────────────────────
        conviction = float(pod_signal.get("conviction", 0))
        if conviction < CONVICTION_DOWNGRADE_THRESHOLD:
            values = {"conviction": conviction, "threshold": CONVICTION_DOWNGRADE_THRESHOLD}
            downgraded = dict(pod_signal)
            downgraded["direction"] = "NEUTRAL"
            downgraded["orchestrator_action"] = OrchestratorDecision.DOWNGRADED_CONVICTION
            downgraded["orchestrator_values"] = values
            _log_decision(market_state, pod_signal, downgraded,
                          OrchestratorDecision.DOWNGRADED_CONVICTION, values)
            return downgraded

        # ─── Pass through ─────────────────────────────────────────────────
        final = dict(pod_signal)
        final["orchestrator_action"] = OrchestratorDecision.PASS_THROUGH
        final["orchestrator_values"] = {"conviction": conviction, "adx_14": adx}
        _log_decision(market_state, pod_signal, final,
                      OrchestratorDecision.PASS_THROUGH, {"conviction": conviction, "adx_14": adx})
        return final

    def process_batch(self, market_states: list) -> list:
        return [self.process(ms) for ms in market_states]
