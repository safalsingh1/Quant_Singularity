"""
finetune_local.py — Local GPU fine-tuning (uses your laptop's GPU)
Same logic as finetune_kaggle.py but auto-detects local CUDA.
Run: python finetune_local.py
"""
import os, sys, json, warnings
# Fix Windows cp1252 encoding error in trl Jinja templates
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path

import mlflow, numpy as np, torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
from trl import SFTTrainer

warnings.filterwarnings("ignore")

# ─── Auto-detect device ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = True  # 4-bit on GPU via bitsandbytes; falls back to fp32 on CPU
print(f"Device: {DEVICE}  |  CUDA available: {torch.cuda.is_available()}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "slm_intern_data"
CLEAN_FILE = DATA_DIR / "finetune_instructions_clean.jsonl"
ADAPTER_DIR = BASE_DIR / "adapter"
ADAPTER_DIR.mkdir(exist_ok=True)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MLFLOW_URI = (BASE_DIR / "mlruns").as_uri()   # file:///C:/... — required on Windows
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("quant-singularity-signal-pod")

# ─── Load data ────────────────────────────────────────────────────────────
if not CLEAN_FILE.exists():
    print("Clean data not found. Run: python data_audit.py first.")
    sys.exit(1)

rows = [json.loads(l) for l in CLEAN_FILE.read_text("utf-8").splitlines() if l.strip()]
print(f"Training rows (post-audit): {len(rows)}")

convictions = [json.loads(r["output"])["conviction"] for r in rows]
print(f"Conviction: mean={np.mean(convictions):.3f} std={np.std(convictions):.3f}")

def format_sample(r):
    return (f"### Instruction:\n{r['instruction']}\n\n"
            f"### Input:\n{r['input']}\n\n"
            f"### Response:\n{r['output']}")

texts = [format_sample(r) for r in rows]
n_val = max(1, int(0.10 * len(texts)))
train_ds = Dataset.from_dict({"text": texts[:-n_val]})
val_ds   = Dataset.from_dict({"text": texts[-n_val:]})
print(f"Train {len(train_ds)}  Val {len(val_ds)}")

# ─── Model loading ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# NOTE: 4-bit quantisation (bitsandbytes) is for INFERENCE only.
# During training with fp16 AMP, bitsandbytes causes a BFloat16/GradScaler conflict
# on RTX 30-series. We train in fp16 without bnb quantisation instead — this still
# fits comfortably in 6GB VRAM for TinyLlama 1.1B.
dtype = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=dtype, device_map="auto" if DEVICE=="cuda" else DEVICE,
)
model.config.use_cache = False

# ─── LoRA config ──────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","v_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ─── Training args ────────────────────────────────────────────────────────
train_args = TrainingArguments(
    output_dir=str(ADAPTER_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2 if DEVICE=="cuda" else 1,
    gradient_accumulation_steps=8,   # effective batch = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=(DEVICE=="cuda"),   # fp16 AMP — works cleanly without bnb during training
    bf16=False,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    optim="adamw_torch",   # avoid bf16-only fused optimisers
)

# ─── Train ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name=f"local_{DEVICE}_r8") as run:
    mlflow.log_params({
        "device": DEVICE,
        "base_model": BASE_MODEL,
        "lora_rank": 8, "lora_alpha": 16,
        "epochs": 3, "lr": 2e-4,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "data_clean_count": len(rows),
    })

    from transformers import DataCollatorForLanguageModeling
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])
    collator  = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model, args=train_args,
        train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=collator,
    )
    result = trainer.train()

    mlflow.log_metrics({
        "train_loss": result.training_loss,
        "train_runtime_s": result.metrics.get("train_runtime", 0),
    })
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            mlflow.log_metric("eval_loss", log["eval_loss"], step=int(log.get("epoch",0)))

    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapter saved: {ADAPTER_DIR}")
    print(f"MLflow run ID: {run.info.run_id}")

print("\nTraining complete. Run: python run_eval.py to evaluate.")
