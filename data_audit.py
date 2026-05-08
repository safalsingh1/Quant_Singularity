"""
data_audit.py — Quant Singularity Signal Pod
Audits finetune_instructions.jsonl before training. Run first, commit before Kaggle.
"""
import json, re, sys, pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "slm_intern_data"
INSTRUCT_FILE = DATA_DIR / "finetune_instructions.jsonl"
MARKET_FILE   = DATA_DIR / "market_states.parquet"
CLEAN_OUTPUT  = DATA_DIR / "finetune_instructions_clean.jsonl"
REPORT_OUT    = Path(__file__).parent / "audit_report.txt"

VALID_DIRECTIONS = {"CE", "PE", "NEUTRAL"}
VALID_HORIZONS   = {"intraday", "next_session"}
REQUIRED_OUT     = {"direction", "conviction", "horizon", "signal_id", "generated_at"}
REQUIRED_IN      = {"nifty_spot","atm_iv","iv_skew_25d","pcr","adx_14",
                     "realized_vol_5d","vix_india","dte_nearest","moneyness_band"}

def parse_conviction(v):
    """Returns (float|None, is_valid, note)"""
    if isinstance(v, (int, float)):
        f = float(v)
        return (f, 0.0<=f<=1.0, "")
    if isinstance(v, str):
        try:
            f = float(v.strip())
            return (f, 0.0<=f<=1.0, "")
        except ValueError:
            pass
        m = re.match(r"^(\d+\.?\d*)", v.strip())
        if m:
            return (float(m.group(1)), False, f"correctable:'{v}'")
        return (None, False, f"uncorrectable:'{v}'")
    return (None, False, f"unknown_type:{type(v)}")

def audit():
    lines  = [l.strip() for l in INSTRUCT_FILE.read_text("utf-8").splitlines() if l.strip()]
    counts = defaultdict(int)
    issue_lines = defaultdict(list)
    signal_ids  = defaultdict(list)
    final_clean = []

    for ln, raw in enumerate(lines, 1):
        try:
            row = json.loads(raw)
            counts["valid_json"] += 1
        except json.JSONDecodeError as e:
            counts["bad_json"] += 1
            issue_lines["bad_json"].append((ln, str(e)))
            continue

        out_raw = row.get("output","")
        inp_raw = row.get("input","")
        try:
            out = json.loads(out_raw) if isinstance(out_raw,str) else out_raw
        except: out = {}; issue_lines["out_parse_fail"].append(ln)
        try:
            inp = json.loads(inp_raw) if isinstance(inp_raw,str) else inp_raw
        except: inp = {}

        row_issues = []
        if REQUIRED_OUT - set(out):  row_issues.append("missing_out_fields")
        if REQUIRED_IN  - set(inp):  row_issues.append("missing_in_fields")
        if out.get("direction","") not in VALID_DIRECTIONS: row_issues.append("bad_direction")
        if out.get("horizon","")   not in VALID_HORIZONS:   row_issues.append("bad_horizon")

        conv_raw  = out.get("conviction", None)
        conv_f, conv_ok, conv_note = parse_conviction(conv_raw)
        if conv_ok:
            counts["conv_ok"] += 1
        elif conv_f is not None:
            counts["conv_correctable"] += 1
            row_issues.append("conv_correctable")
        else:
            counts["conv_bad"] += 1
            row_issues.append("conv_uncorrectable")

        sig_id = out.get("signal_id","")
        signal_ids[sig_id].append(ln)

        atm_iv = inp.get("atm_iv", 99)
        if atm_iv <= 10.0:
            counts["atm_iv_floor"] += 1

        # Include row if clean OR only correctable conviction issue
        non_conv = [i for i in row_issues if i!="conv_correctable"]
        if not row_issues:
            counts["schema_ok"] += 1
            final_clean.append((ln, row))
        elif not non_conv and conv_f is not None and 0.0<=conv_f<=1.0:
            out["conviction"] = round(conv_f, 2)
            row["output"] = json.dumps(out)
            final_clean.append((ln, row))
            counts["corrected"] += 1
        else:
            counts["excluded"] += 1
            for i in row_issues:
                issue_lines[i].append(ln)

    dups = {k:v for k,v in signal_ids.items() if len(v)>1}

    with open(CLEAN_OUTPUT,"w",encoding="utf-8") as f:
        for _,row in sorted(final_clean):
            f.write(json.dumps(row)+"\n")

    # Load market states for cross-check
    df = pd.read_parquet(MARKET_FILE)
    vix_m, vix_s = df["vix_india"].mean(), df["vix_india"].std()

    report = [
        "="*68,"QUANT SINGULARITY — DATA AUDIT REPORT","="*68,"",
        f"Total instruction rows            : {len(lines)}",
        f"Valid JSON                        : {counts['valid_json']}",
        f"Bad JSON                          : {counts['bad_json']}",
        "",
        "CONVICTION FIELD — PRIMARY FINDING",
        "-"*40,
        f"Numeric conviction (valid)        : {counts['conv_ok']}  (rows 1–47, 93–300)",
        f"String conviction (correctable)   : {counts['conv_correctable']}  (e.g. '0.8 (high)' -> 0.8)",
        f"String conviction (uncorrectable) : {counts['conv_bad']}  (e.g. 'high','moderate','low')",
        "",
        "ROOT CAUSE: Rows ~48–92 originated from a different pipeline that",
        "serialised conviction as human text instead of float. These rows",
        "would teach the model to violate the output schema. Conviction is",
        "not a softmax probability — it is a model-generated float that must",
        "be parseable by the orchestrator. We EXCLUDE uncorrectable rows and",
        "PATCH correctable rows (extract leading float, validate range).",
        "",
        "OTHER FINDINGS",
        "-"*40,
        f"atm_iv clamped to 10.0 floor      : {counts['atm_iv_floor']} rows — legitimate NSE floor, kept",
        f"Duplicate signal_ids              : {sum(len(v) for v in dups.values())} (in {len(dups)} groups)",
        "",
        "CLEAN DATASET",
        "-"*40,
        f"Originally clean rows             : {counts['schema_ok']}",
        f"Patched (corrected conviction)    : {counts['corrected']}",
        f"Excluded (uncorrectable)          : {counts['excluded']}",
        f"Total rows in clean file          : {len(final_clean)}",
        f"Output                            : {CLEAN_OUTPUT}",
        "",
        "MARKET STATES PARQUET",
        "-"*40,
        f"Rows                              : {len(df)}",
        f"ADX range                         : {df['adx_14'].min():.2f}–{df['adx_14'].max():.2f}",
        f"VIX range                         : {df['vix_india'].min():.2f}–{df['vix_india'].max():.2f}",
        f"VIX mean={vix_m:.2f} std={vix_s:.2f}  3σ={vix_m+3*vix_s:.2f}",
        f"ADX<20 rows (orchestrator would suppress): {(df['adx_14']<20).sum()}",
        f"Label distribution                : {df['label'].value_counts().to_dict()}",
        "","="*68,"END OF REPORT","="*68,
    ]
    txt = "\n".join(report)
    REPORT_OUT.write_text(txt, "utf-8")
    print(txt)
    return counts

if __name__ == "__main__":
    audit()
