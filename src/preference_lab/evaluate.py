"""Shared frozen evaluator. Every model (base, SFT, DPO, GRPO) is run through this same code with the same decoding.

  python -m preference_lab.evaluate --split val --name base [--adapter runs/sft/adapter] [--limit 20]

Writes runs/<name>-<split>/predictions.jsonl (EVERY row, failures included), metrics.json and usage.json.
Decoding is greedy, max 96 new tokens, so the same checkpoint on the same prompts gives the same answers on this machine.
"""
from __future__ import annotations
import argparse, collections, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .oracle import verify
from .splits import load

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_NEW = 96
BATCH = 16


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(adapter: str | None = None):
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model, dev


def chat(tok, row) -> str:
    msgs = [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["prompt"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(tok, model, dev, rows, max_new=MAX_NEW, batch=BATCH):
    outs, n_new = [], 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([chat(tok, r) for r in chunk], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for row_ids in new:
            n_new += int((row_ids != tok.pad_token_id).sum())
        outs += tok.batch_decode(new, skip_special_tokens=True)
    return outs, n_new


def score(rows, outs) -> list[dict]:
    res = []
    for r, o in zip(rows, outs):
        v = verify(o, r["label"], r["required_ids"])
        res.append({"row_id": r["row_id"], "group_id": r["group_id"], "split": r["split"], "template": r["template"],
                    "novel_values": r["novel_values"], "rule": r["rule"], "label": r["label"], "required_ids": r["required_ids"],
                    "output": o, **v})
    return res


def metrics(preds: list[dict]) -> dict:
    def block(ps):
        n = len(ps)
        if n == 0: return {"n": 0}
        return {"n": n, "correct": sum(p["correct"] for p in ps), "correct_rate": sum(p["correct"] for p in ps) / n,
                "valid_rate": sum(p["valid"] for p in ps) / n, "format_rate": sum(p["format_ok"] for p in ps) / n, "action_rate": sum(p["action_ok"] for p in ps) / n,
                "evidence_rate": sum(p["evidence_ok"] for p in ps) / n}
    m = {"overall": block(preds)}
    for lab in ("ACCEPT", "DENY", "HOLD"):
        m[f"label_{lab}"] = block([p for p in preds if p["label"] == lab])
    for t in sorted({p["template"] for p in preds}):
        m[f"template_{t}"] = block([p for p in preds if p["template"] == t])
    m["novel_values"] = block([p for p in preds if p["novel_values"]])
    m["seen_values"] = block([p for p in preds if not p["novel_values"]])
    m["rules"] = {r: block([p for p in preds if p["rule"] == r]) for r in sorted({p["rule"] for p in preds})}
    # predicted-action distribution shows collapse (for example always HOLD)
    m["predicted_actions"] = dict(collections.Counter(str(p["action"]) for p in preds))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--name", required=True)
    ap.add_argument("--adapter"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = load(a.split)
    if a.limit: rows = rows[: a.limit]
    tok, model, dev = load_model(a.adapter)
    t0 = time.time()
    outs, n_new = generate(tok, model, dev, rows)
    dt = time.time() - t0
    preds = score(rows, outs)
    out = Path("runs") / f"{a.name}-{a.split}"; out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    m = metrics(preds)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    usage = {"rows": len(rows), "new_tokens": n_new, "seconds": round(dt, 1), "per_row_s": round(dt / len(rows), 3), "device": dev,
             "model": MODEL, "revision": REVISION, "adapter": a.adapter, "max_new": MAX_NEW, "batch": BATCH,
             "mps_peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev == "mps" else None}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    print(json.dumps({"split": a.split, "name": a.name, "overall": m["overall"], "predicted_actions": m["predicted_actions"], "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
