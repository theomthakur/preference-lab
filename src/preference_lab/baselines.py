"""Stronger baselines with NO training: worked examples in the prompt (few-shot), and larger instruction models.

  python -m preference_lab.baselines --model Qwen/Qwen2.5-0.5B-Instruct --k 3 --name fs-0.5b-k3
  python -m preference_lab.baselines --model Qwen/Qwen2.5-1.5B-Instruct --k 3 --name fs-1.5b-k3 --dtype bfloat16

Few-shot examples come from the SFT split only (never validation or test), are chosen deterministically (one per label first, with
distinct rules, seed 7) and are listed in config.json. Scoring, decoding and metrics are the shared ones from evaluate.py, run on the
validation split. Writes runs/<name>-val/{predictions.jsonl, metrics.json, errors.json, usage.json, config.json}.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import answer_json
from .errors import breakdown
from .evaluate import MAX_NEW, device, metrics, score
from .splits import load


def pick_examples(k: int, seed: int = 7) -> list[dict]:
    """Deterministic worked examples from the SFT split: cycle through the three labels, never repeating a rule until all are used."""
    rows = load("sft"); random.Random(seed).shuffle(rows)
    chosen, used_rules, labels = [], set(), ["ACCEPT", "DENY", "HOLD"]
    for want in (labels * ((k + 2) // 3))[:k]:
        for r in rows:
            if r["label"] == want and r["rule"] not in used_rules and r not in chosen and len(r["required_ids"]) >= 2:
                chosen.append(r); used_rules.add(r["rule"]); break
        else:
            for r in rows:
                if r["label"] == want and r not in chosen: chosen.append(r); break
    return chosen


def messages(row: dict, examples: list[dict]) -> list[dict]:
    m = [{"role": "system", "content": row["system"]}]
    for e in examples:
        m += [{"role": "user", "content": e["prompt"]}, {"role": "assistant", "content": answer_json(e["label"], e["required_ids"])}]
    return m + [{"role": "user", "content": row["prompt"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--k", type=int, default=3); ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"]); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int); ap.add_argument("--split", default="val")
    a = ap.parse_args()
    assert a.split == "val", "baselines are evaluated on validation only; the held-out test is not used to compare setups"
    dev = device(); rev = HfApi().model_info(a.model).sha
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, dtype=getattr(torch, a.dtype)).to(dev).eval()
    rows = load(a.split)[: a.limit] if a.limit else load(a.split)
    ex = pick_examples(a.k)
    outs, n_new, t0 = [], 0, time.time()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        enc = tok([tok.apply_chat_template(messages(r, ex), tokenize=False, add_generation_prompt=True) for r in chunk], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        n_new += int((new != tok.pad_token_id).sum()); outs += tok.batch_decode(new, skip_special_tokens=True)
        if (i // a.batch) % 10 == 0: print(f"{i + len(chunk)}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    preds = score(rows, outs); out = Path("runs") / f"{a.name}-{a.split}"; out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    m = metrics(preds); (out / "metrics.json").write_text(json.dumps(m, indent=2))
    (out / "errors.json").write_text(json.dumps(breakdown(preds, {r["row_id"]: r for r in rows}), indent=1))
    (out / "config.json").write_text(json.dumps({"model": a.model, "revision": rev, "dtype": a.dtype, "k": a.k, "example_row_ids": [e["row_id"] for e in ex],
        "example_rules": [e["rule"] for e in ex], "example_labels": [e["label"] for e in ex], "decoding": "greedy", "max_new": MAX_NEW, "split": a.split, "training": "none"}, indent=2))
    usage = {"rows": len(rows), "seconds": round(dt, 1), "per_row_s": round(dt / len(rows), 2), "new_tokens": n_new, "device": dev,
             "mps_peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev == "mps" else None}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    o = m["overall"]; print(json.dumps({"name": a.name, "strict": f"{o['correct']}/{o['n']}", "action_rate": round(o["action_rate"], 3), "evidence_rate": round(o["evidence_rate"], 3), "format_rate": round(o["format_rate"], 3), "usage": usage}))


if __name__ == "__main__":
    main()
