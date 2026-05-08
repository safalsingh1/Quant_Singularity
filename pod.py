"""
pod.py — Signal Pod (inference only)
Loads the fine-tuned TinyLlama adapter and produces structured JSON signals.
CPU inference, 4-bit quantization. Fallback to NEUTRAL on any failure.
"""
import json, uuid, logging, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger("signal_pod")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

ADAPTER_PATH = Path(__file__).parent / "adapter"
BASE_MODEL    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
VALID_DIRECTIONS = {"CE","PE","NEUTRAL"}
VALID_HORIZONS   = {"intraday","next_session"}


def _build_prompt(market_state: dict, retrieved_episodes: Optional[list] = None) -> str:
    ms = market_state
    feature_str = (
        f"nifty_spot={ms.get('nifty_spot',0):.2f}, "
        f"atm_iv={ms.get('atm_iv',0):.4f}, "
        f"iv_skew_25d={ms.get('iv_skew_25d',0):.4f}, "
        f"pcr={ms.get('pcr',0):.4f}, "
        f"adx_14={ms.get('adx_14',0):.2f}, "
        f"realized_vol_5d={ms.get('realized_vol_5d',0):.4f}, "
        f"vix_india={ms.get('vix_india',0):.2f}, "
        f"dte_nearest={ms.get('dte_nearest',0)}, "
        f"moneyness_band={ms.get('moneyness_band','ATM')}"
    )

    rag_block = ""
    if retrieved_episodes:
        parts = []
        for ep in retrieved_episodes[:3]:
            parts.append(
                f"  [{ep.get('episode_id','?')}] Regime: {ep.get('regime','?')} | "
                f"ADX={ep['market_state'].get('adx_14','?')}, "
                f"VIX={ep['market_state'].get('vix_india','?')}, "
                f"PCR={ep['market_state'].get('pcr','?')} | "
                f"Outcome: {ep.get('outcome','?')} — {ep.get('outcome_description','')}"
            )
        rag_block = "\n\nRelevant historical episodes:\n" + "\n".join(parts)

    instruction = (
        "You are a trading signal generator for NIFTY 50 options. "
        "Analyze the provided market state snapshot and generate a structured trading signal. "
        "Return ONLY valid JSON matching the required schema. "
        'Schema: {"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, '
        '"horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}'
    )
    input_str = json.dumps(
        {k: ms.get(k) for k in
         ["nifty_spot","atm_iv","iv_skew_25d","pcr","adx_14",
          "realized_vol_5d","vix_india","dte_nearest","moneyness_band"]}
    )
    prompt = f"### Instruction:\n{instruction}{rag_block}\n\n### Input:\n{input_str}\n\n### Response:\n"
    return prompt


def _make_fallback(reason: str) -> dict:
    return {
        "direction": "NEUTRAL",
        "conviction": 0.0,
        "horizon": "intraday",
        "signal_id": str(uuid.uuid5(uuid.NAMESPACE_URL, reason + datetime.now().isoformat())),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "_fallback": True,
        "_fallback_reason": reason,
    }


class SignalPod:
    """
    Wraps TinyLlama + LoRA adapter for NIFTY options signal generation.
    CPU-only inference with 4-bit quantization via bitsandbytes.
    """

    def __init__(self, adapter_path: Optional[str] = None, use_4bit: bool = True, device: str = "auto"):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._adapter_path = adapter_path or str(ADAPTER_PATH)
        self._use_4bit = use_4bit
        self._device = device

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            logger.info("Loading tokenizer from %s", BASE_MODEL)
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            if self._use_4bit and self._device != "cuda":
                # 4-bit on CPU via bitsandbytes
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Loading base model with 4-bit quantization")
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, quantization_config=bnb_config,
                    device_map="auto", trust_remote_code=True,
                )
            elif self._device == "cuda":
                logger.info("Loading base model on GPU")
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.float16,
                    device_map="cuda", trust_remote_code=True,
                )
            else:
                logger.info("Loading base model in fp32 on CPU (no quantization)")
                base = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL, torch_dtype=torch.float32,
                    device_map="cpu",
                )

            if Path(self._adapter_path).exists():
                logger.info("Loading LoRA adapter from %s", self._adapter_path)
                self._model = PeftModel.from_pretrained(base, self._adapter_path)
            else:
                logger.warning("Adapter not found at %s — using base model only", self._adapter_path)
                self._model = base

            self._model.eval()
            self._loaded = True
            logger.info("SignalPod loaded OK")
        except Exception as e:
            logger.error("SignalPod load failed: %s", e)
            self._loaded = False

    def predict(self, market_state: dict, retrieved_episodes: Optional[list] = None,
                max_new_tokens: int = 128, temperature: float = 0.1) -> dict:
        """
        Generate a structured trading signal for the given market state.
        Returns parsed dict with keys: direction, conviction, horizon, signal_id, generated_at.
        On any failure returns NEUTRAL / conviction=0.0 fallback.
        """
        self._load()
        if not self._loaded:
            return _make_fallback("model_not_loaded")

        prompt = _build_prompt(market_state, retrieved_episodes)

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[-1]:]
            raw_text  = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error("Inference error: %s", e)
            return _make_fallback(f"inference_error:{e}")

        return self._parse_output(raw_text, market_state)

    def _parse_output(self, raw_text: str, market_state: dict) -> dict:
        """
        Extract and validate JSON from model output.
        Handles two failure modes:
          1. Model wraps output in markdown fences -> strip and retry
          2. Long signal_id truncates JSON before closing brace -> regex extraction
        """
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?\s*", "", raw_text).strip()

        # Try 1: direct JSON parse on full text or largest braced block
        candidates = [clean, raw_text]
        candidates += re.findall(r'\{[^{}]*\}', clean, re.DOTALL)
        candidates += re.findall(r'\{[^{}]*\}', raw_text, re.DOTALL)
        for candidate in candidates:
            try:
                obj = json.loads(candidate.strip())
                return self._validate(obj, raw_text)
            except (json.JSONDecodeError, ValueError):
                continue

        # Try 2: regex extraction from TRUNCATED JSON
        # The model knows direction/conviction/horizon but signal_id overflows the budget.
        # Extract the three mandatory fields individually and generate a fresh UUID.
        dir_m  = re.search(r'"direction"\s*:\s*"(CE|PE|NEUTRAL)"', raw_text)
        conv_m = re.search(r'"conviction"\s*:\s*(\d*\.?\d+)', raw_text)
        horiz_m = re.search(r'"horizon"\s*:\s*"(intraday|next_session)"', raw_text)

        if dir_m and conv_m and horiz_m:
            obj = {
                "direction":  dir_m.group(1),
                "conviction": float(conv_m.group(1)),
                "horizon":    horiz_m.group(1),
                "signal_id":  str(uuid.uuid4()),   # always generate fresh UUID
                "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "_recovered_from_truncation": True,
            }
            logger.info("Recovered signal from truncated JSON: dir=%s conv=%s",
                        obj["direction"], obj["conviction"])
            return self._validate(obj, raw_text)

        logger.warning("Pod could not parse output: %s", raw_text[:200])
        return _make_fallback(f"parse_failure: {raw_text[:100]}")

    def _validate(self, obj: dict, raw_text: str) -> dict:
        """Validate schema and normalise conviction."""
        direction = obj.get("direction", "")
        if direction not in VALID_DIRECTIONS:
            return _make_fallback(f"invalid_direction:'{direction}'")

        horizon = obj.get("horizon", "")
        if horizon not in VALID_HORIZONS:
            return _make_fallback(f"invalid_horizon:'{horizon}'")

        # Parse conviction
        conv_raw = obj.get("conviction", None)
        try:
            conv = float(conv_raw)
            if not (0.0 <= conv <= 1.0):
                raise ValueError(f"out_of_range:{conv}")
        except (ValueError, TypeError) as e:
            return _make_fallback(f"invalid_conviction:{conv_raw}:{e}")

        # Generate signal_id if missing
        sig_id = obj.get("signal_id") or str(uuid.uuid4())
        gen_at = obj.get("generated_at") or datetime.now(timezone.utc).astimezone().isoformat()

        return {
            "direction":   direction,
            "conviction":  round(conv, 4),
            "horizon":     horizon,
            "signal_id":   sig_id,
            "generated_at": gen_at,
            "_raw_text":   raw_text[:300],
        }


# Module-level convenience
_default_pod: Optional[SignalPod] = None

def get_pod(adapter_path: Optional[str] = None, use_4bit: bool = True, device: str = "auto") -> SignalPod:
    global _default_pod
    if _default_pod is None:
        _default_pod = SignalPod(adapter_path=adapter_path, use_4bit=use_4bit, device=device)
    return _default_pod
