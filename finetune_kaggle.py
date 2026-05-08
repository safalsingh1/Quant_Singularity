"""
finetune_kaggle.py
==================
TinyLlama-1.1B fine-tuning notebook (Python script form for Kaggle).
- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Fine-tuning: LoRA (rank=8, alpha=16, dropout=0.05) via PEFT
- Training data: finetune_instructions_clean.jsonl (post-audit)
- Inference: 4-bit quantised, CPU-compatible
- Tracking: MLflow from run 1

Run in Kaggle with T4 GPU. GPU also works locally.
Kaggle notebook URL: https://www.kaggle.com/code/safalsingh/quant-singularity-finetune-and-eval

Notes:
------
Conviction design:
  The conviction field is NOT a softmax probability over direction tokens.
  Softmax(next_token) gives P(token | prefix), not a calibrated confidence
  in the signal's profitability. Instead we:
  1. Include conviction as a numeric value in the supervised training target.
  2. The model learns to generate a float by observing training data that
     labels conviction proportional to how many directional features agree.
  3. Post-training, conviction is the numeric value the model GENERATES
     as text — we parse it and validate 0.0–1.0. It is a learned proxy
     for how aligned the market state is with the label pattern.
  4. We analyse conviction reliability via ECE (Expected Calibration Error):
     does higher conviction correlate with higher directional accuracy?
"""

# ─── Cell 0: Install dependencies ─────────────────────────────────────────
import subprocess, sys
def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

pip(
    "transformers>=4.40",
    "peft>=0.9",
    "trl>=0.8",
    "bitsandbytes>=0.43",
    "accelerate>=0.27",
    "datasets>=2.18",
    "mlflow>=2.11",
    "scipy",
)

# ─── Cell 1: Imports ──────────────────────────────────────────────────────
import json, math, os, uuid, warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from trl import SFTTrainer

warnings.filterwarnings("ignore")

# ─── Cell 2: Paths & config ───────────────────────────────────────────────
# In Kaggle: upload slm_intern_data as a dataset.
# Locally:   set DATA_DIR to slm_intern_data/
DATA_DIR = Path("/kaggle/input/datasets/safalsingh/quant-singularity-data")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/input/quant-singularity-data")
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).parent / "slm_intern_data"

OUTPUT_DIR    = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
ADAPTER_DIR   = OUTPUT_DIR / "adapter"
ADAPTER_DIR.mkdir(exist_ok=True)

CLEAN_JSONL   = DATA_DIR / "finetune_instructions_clean.jsonl"
MARKET_PARQUET = DATA_DIR / "market_states.parquet"

BASE_MODEL    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MLFLOW_URI    = str(OUTPUT_DIR / "mlruns")
EXPERIMENT    = "quant-singularity-signal-pod"

# ─── Cell 3: MLflow setup ─────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
print(f"MLflow tracking: {MLFLOW_URI}")

# ─── Cell 4: Load & inspect training data ─────────────────────────────────
rows  = [json.loads(l) for l in CLEAN_JSONL.read_text("utf-8").splitlines() if l.strip()]
print(f"Clean training rows: {len(rows)}")

# Verify conviction distribution
convictions = [json.loads(r["output"])["conviction"] for r in rows]
print(f"Conviction stats: mean={np.mean(convictions):.3f} std={np.std(convictions):.3f} "
      f"min={min(convictions):.2f} max={max(convictions):.2f}")

# Direction distribution
directions  = [json.loads(r["output"])["direction"] for r in rows]
from collections import Counter
print(f"Direction distribution: {Counter(directions)}")

# ─── Cell 5: Dataset formatting ───────────────────────────────────────────
def format_sample(row: dict) -> str:
    """Alpaca-style instruction format."""
    return (
        f"### Instruction:\n{row['instruction']}\n\n"
        f"### Input:\n{row['input']}\n\n"
        f"### Response:\n{row['output']}"
    )

texts = [format_sample(r) for r in rows]

# Train / val split (90/10) — note this is within training window only
n_val = max(1, int(0.10 * len(texts)))
val_texts   = texts[-n_val:]
train_texts = texts[:-n_val]
print(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

train_ds = Dataset.from_dict({"text": train_texts})
val_ds   = Dataset.from_dict({"text": val_texts})

# ─── Cell 6: Quantisation config ─────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ─── Cell 7: Load base model ──────────────────────────────────────────────
print(f"Loading {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False

# ─── Cell 8: LoRA config ──────────────────────────────────────────────────
# Rank 8 chosen: 4–16 range. Justification:
#   - Task: structured JSON generation from 9 numeric features (small domain).
#   - rank=4 under-fits; rank=16 risks overfitting on 261 samples.
#   - rank=8 balances expressiveness with regularisation on small data.
#   - alpha=16 (2×rank) gives stable gradient scale.
#   - dropout=0.05 light regularisation (data is small).
#   - Target modules: q_proj, v_proj (standard for TinyLlama MLP attention).
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","v_proj"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ─── Cell 9: Training arguments ──────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch 16
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",   # MLflow handled manually
    dataloader_num_workers=2,
    max_grad_norm=1.0,
)

# ─── Cell 10: Train with MLflow ──────────────────────────────────────────
with mlflow.start_run(run_name="tinyllama_lora_r8_base") as run:
    mlflow.log_params({
        "base_model":        BASE_MODEL,
        "lora_rank":         8,
        "lora_alpha":        16,
        "lora_dropout":      0.05,
        "target_modules":    "q_proj,v_proj",
        "epochs":            3,
        "lr":                2e-4,
        "batch_size_eff":    16,
        "train_samples":     len(train_texts),
        "val_samples":       len(val_texts),
        "data_audit_clean":  len(rows),
        "data_excluded":     300 - len(rows),
        "conviction_mean":   round(float(np.mean(convictions)),4),
    })

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    result = trainer.train()

    mlflow.log_metrics({
        "train_loss":  result.training_loss,
        "train_runtime_s": result.metrics.get("train_runtime",0),
    })
    for epoch_log in trainer.state.log_history:
        if "eval_loss" in epoch_log:
            mlflow.log_metric("eval_loss", epoch_log["eval_loss"],
                              step=int(epoch_log.get("epoch",0)))

    # Save adapter
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved to {ADAPTER_DIR}")
    mlflow.log_artifact(str(ADAPTER_DIR), artifact_path="adapter")
    print(f"MLflow run ID: {run.info.run_id}")

# ─── Cell 11: Quick schema compliance smoke test ──────────────────────────
print("\n=== SCHEMA COMPLIANCE SMOKE TEST ===")
from peft import PeftModel as PM

test_model = PM.from_pretrained(base_model, str(ADAPTER_DIR))
test_model.eval()

VALID_DIRS = {"CE","PE","NEUTRAL"}
VALID_HORIZ = {"intraday","next_session"}

def smoke_test(ms_dict: dict) -> dict:
    prompt = (
        "### Instruction:\n"
        "You are a trading signal generator for NIFTY 50 options. "
        "Analyze the provided market state snapshot and generate a structured trading signal. "
        "Return ONLY valid JSON matching the required schema. "
        '{"direction": "CE"|"PE"|"NEUTRAL", "conviction": float 0.0-1.0, '
        '"horizon": "intraday"|"next_session", "signal_id": string, "generated_at": string}\n\n'
        f"### Input:\n{json.dumps(ms_dict)}\n\n### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(test_model.device)
    with torch.no_grad():
        out = test_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=True,
                                  pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    try:
        obj = json.loads(text)
        assert obj["direction"] in VALID_DIRS
        assert 0.0 <= float(obj["conviction"]) <= 1.0
        assert obj["horizon"] in VALID_HORIZ
        return {"ok": True, "text": text}
    except Exception as e:
        return {"ok": False, "text": text, "err": str(e)}

test_states = [
    {"nifty_spot":22859.61,"atm_iv":13.41,"iv_skew_25d":3.88,"pcr":1.13,"adx_14":29.35,
     "realized_vol_5d":13.6,"vix_india":14.08,"dte_nearest":2,"moneyness_band":"ATM"},
    {"nifty_spot":21800.00,"atm_iv":10.0,"iv_skew_25d":1.5,"pcr":0.87,"adx_14":15.0,
     "realized_vol_5d":8.0,"vix_india":18.0,"dte_nearest":1,"moneyness_band":"1pct_OTM"},
]
n_ok = 0
for i, ms in enumerate(test_states):
    r = smoke_test(ms)
    status = "PASS" if r["ok"] else "FAIL"
    print(f"Test {i+1}: {status}  output={r['text'][:80]}")
    if r["ok"]: n_ok += 1
print(f"Smoke test pass rate: {n_ok}/{len(test_states)}")

with mlflow.start_run(run_name="smoke_test", nested=False):
    mlflow.log_metric("smoke_pass_rate", n_ok/len(test_states))
