"""Run the frozen held-out TEST set once, for every model, with the same decoding as validation.

  python -m preference_lab.final_test          # refuses to run if runs/test-results exists (the test is used once)

Outputs: runs/test-results/{predictions_<model>.jsonl, test_metrics.json, intervals.json, FREEZE.json}.
Intervals: group-level bootstrap (resample whole groups, because each group appears in two rendered rows), 2,000 resamples, seed 7.
"""
from __future__ import annotations
import json, random, subprocess, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from . import evaluate as E
from .splits import load

OUT = Path("runs/test-results")
MODELS = {
    "base": ("base", None, None),
    "sft": ("base", "runs/sft/adapter", None),
    "dpo_clean": ("base", "runs/dpo-clean/adapter", None),
    "dpo_corrupt": ("base", "runs/dpo-corrupt/adapter", None),
    "grpo": ("runs/sft-merged", "runs/grpo/adapter", None),
}


def sha(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")) if Path(path).is_dir() else [Path(path)]:
        if p.is_file():
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_model(name, base_src, adapter, rows):
    tok = AutoTokenizer.from_pretrained(base_src if base_src != "base" else E.MODEL, revision=None if base_src != "base" else E.REVISION, padding_side="left")
    dev = E.device()
    base = AutoModelForCausalLM.from_pretrained(E.MODEL if base_src == "base" else base_src,
                                                revision=E.REVISION if base_src == "base" else None, dtype=torch.float32).to(dev)
    model = PeftModel.from_pretrained(base, adapter).eval() if adapter else base.eval()
    t0 = time.time()
    outs, n_new = E.generate(tok, model, dev, rows)
    dt = time.time() - t0
    preds = E.score(rows, outs)
    (OUT / f"predictions_{name}.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    return preds, {"seconds": round(dt, 1), "new_tokens": n_new, "rows": len(rows)}


def group_bootstrap(a: dict, b: dict | None, groups: dict, n=2000, seed=7):
    """a, b: row_id -> 0/1. Returns the strict-accuracy 95% interval, and the paired difference interval if b is given."""
    r = random.Random(seed); gids = list(groups)
    rows_of = {g: groups[g] for g in gids}
    acc, diffs = [], []
    for _ in range(n):
        pick = [r.choice(gids) for _ in gids]
        rs = [rid for g in pick for rid in rows_of[g]]
        acc.append(sum(a[x] for x in rs) / len(rs))
        if b is not None:
            diffs.append(sum(a[x] - b[x] for x in rs) / len(rs))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    res = {"point": round(sum(a.values()) / len(a), 4), "ci95": [round(q(acc, .025), 4), round(q(acc, .975), 4)]}
    if b is not None:
        res["paired_diff_point"] = round((sum(a.values()) - sum(b.values())) / len(a), 4)
        res["paired_diff_ci95"] = [round(q(diffs, .025), 4), round(q(diffs, .975), 4)]
    return res


def main():
    if OUT.exists():
        raise SystemExit("runs/test-results already exists: the held-out test has been used. Do not rerun it.")
    OUT.mkdir(parents=True)
    rows = load("test")
    groups = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r["row_id"])
    freeze = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "test_file_sha": sha("data/test.jsonl"), "adapters": {}, "decoding": {"greedy": True, "max_new": E.MAX_NEW}}
    for k, (_, ad, _) in MODELS.items():
        if ad: freeze["adapters"][k] = sha(ad)
    (OUT / "FREEZE.json").write_text(json.dumps(freeze, indent=2))
    allp, usage, correct = {}, {}, {}
    for name, (base_src, adapter, _) in MODELS.items():
        preds, u = run_model(name, base_src, adapter, rows)
        usage[name] = u
        correct[name] = {p["row_id"]: int(p["correct"]) for p in preds}
        allp[name] = preds
        print(name, "strict", round(sum(correct[name].values()) / len(rows), 3), u, flush=True)
    seen = {r["row_id"] for r in rows if r["template"] in (0, 1, 2)}
    held = {r["row_id"] for r in rows if r["template"] == 3}
    summary = {}
    for name in MODELS:
        c = correct[name]
        summary[name] = {"all_rows": group_bootstrap(c, None, groups),
                         "seen_template": group_bootstrap({k: v for k, v in c.items() if k in seen}, None, groups),
                         "held_out_template": group_bootstrap({k: v for k, v in c.items() if k in held}, None, groups),
                         "usage": usage[name]}
    diffs = {"dpo_clean_vs_sft": ("dpo_clean", "sft"), "dpo_corrupt_vs_dpo_clean": ("dpo_corrupt", "dpo_clean"),
             "grpo_vs_sft": ("grpo", "sft"), "sft_vs_base": ("sft", "base")}
    paired = {k: group_bootstrap(correct[a], correct[b], groups) for k, (a, b) in diffs.items()}
    (OUT / "test_metrics.json").write_text(json.dumps({"models": summary, "paired": paired}, indent=2))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
