"""Build the read-only dashboard data from the committed synthetic test cases and the saved predictions.

  python3 web/generate_data.py             # validate everything, then write web/data.json
  python3 web/generate_data.py --check     # validate and confirm web/data.json is byte-identical; writes nothing
  python3 web/generate_data.py --output X  # validate, then write to X instead

This script never runs a model and never writes to data/ or runs/. It only reads them. It refuses to build if the saved files
do not line up, because a dashboard that joins mismatched rows would quietly show the wrong evidence.

Checks, for every model and every test case:
  * each case appears exactly once in the predictions (no missing, extra or duplicate row ids);
  * the prediction row carries the same group_id, template, label, rule, split, novel_values and required_ids as the source case;
  * the saved flags agree with each other (correct == action_ok and evidence_ok);
  * the accuracy recomputed from the predictions equals the saved test_metrics.json value exactly, for all rows, for the seen
    wording and for the held-out wording, and the saved paired differences equal the recomputed ones.
The output records a SHA-256 of every source file, so a stale data.json is detectable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("base", "sft", "dpo_clean", "dpo_corrupt", "grpo")
VIEWS = ("all_rows", "seen_template", "held_out_template")
HELD_OUT_TEMPLATE = 3
PAIRS = {"sft_vs_base": ("sft", "base"), "dpo_clean_vs_sft": ("dpo_clean", "sft"),
         "dpo_corrupt_vs_dpo_clean": ("dpo_corrupt", "dpo_clean"), "grpo_vs_sft": ("grpo", "sft")}
CASE_KEYS = ("row_id", "group_id", "rule", "prompt", "label", "required_ids", "template", "novel_values")
PRED_KEYS = ("output", "action", "valid", "format_ok", "action_ok", "evidence_ok", "correct")
SAME_AS_CASE = ("group_id", "template", "label", "rule", "required_ids", "novel_values")


class DataError(ValueError):
    """The saved files do not line up, so no dashboard data is built."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DataError(f"missing file: {path}")
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            raise DataError(f"{path.name} line {n} is not valid JSON ({e})")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(root: Path, model: str, cases: dict) -> dict[str, dict]:
    path = root / f"runs/test-results/predictions_{model}.jsonl"
    rows = read_jsonl(path)
    by_id, dupes = {}, []
    for r in rows:
        rid = r.get("row_id")
        if rid in by_id:
            dupes.append(rid)
        by_id[rid] = r
    if dupes:
        raise DataError(f"{model}: duplicate prediction rows, for example {dupes[:3]}")
    missing, extra = sorted(set(cases) - set(by_id)), sorted(set(by_id) - set(cases))
    if missing:
        raise DataError(f"{model}: {len(missing)} cases have no prediction, for example {missing[:3]}")
    if extra:
        raise DataError(f"{model}: {len(extra)} predictions match no case, for example {extra[:3]}")
    for rid, p in by_id.items():
        c = cases[rid]
        if p.get("split") != "test":
            raise DataError(f"{model}: row {rid} has split {p.get('split')!r}, expected 'test'")
        for key in SAME_AS_CASE:
            if p.get(key) != c.get(key):
                raise DataError(f"{model}: row {rid} is misaligned on '{key}' (prediction {p.get(key)!r}, case {c.get(key)!r})")
        for key in ("format_ok", "action_ok", "evidence_ok", "correct", "valid"):
            if not isinstance(p.get(key), bool):
                raise DataError(f"{model}: row {rid} has a non-boolean '{key}': {p.get(key)!r}")
        if p["correct"] != (p["action_ok"] and p["evidence_ok"]):
            raise DataError(f"{model}: row {rid} has inconsistent flags (correct={p['correct']}, action_ok={p['action_ok']}, evidence_ok={p['evidence_ok']})")
        if not isinstance(p.get("output"), str):
            raise DataError(f"{model}: row {rid} has no text output")
    return by_id


def check_metrics(metrics: dict, cases: dict, preds: dict) -> None:
    """Recompute every saved accuracy and paired difference from the predictions. Exact match, rounded to 4 places like the saved file."""
    held = {rid for rid, c in cases.items() if c["template"] == HELD_OUT_TEMPLATE}
    subsets = {"all_rows": set(cases), "seen_template": set(cases) - held, "held_out_template": held}
    counts = {}
    for m in MODELS:
        if m not in metrics.get("models", {}):
            raise DataError(f"test_metrics.json has no entry for {m}")
        for view, ids in subsets.items():
            k = sum(preds[m][rid]["correct"] for rid in ids)
            counts[(m, view)] = k
            saved = metrics["models"][m][view]
            if saved["n_rows"] != len(ids):
                raise DataError(f"{m}/{view}: saved n_rows {saved['n_rows']} but the predictions have {len(ids)}")
            if round(k / len(ids), 4) != saved["point"]:
                raise DataError(f"{m}/{view}: saved accuracy {saved['point']} but the predictions give {round(k / len(ids), 4)}")
    n = len(cases)
    for key, (a, b) in PAIRS.items():
        saved = metrics.get("paired", {}).get(key)
        if saved is None:
            raise DataError(f"test_metrics.json has no paired entry '{key}'")
        # Same arithmetic as src/preference_lab/test_analysis.py: subtract the two unrounded accuracies, then round to 4 places.
        # (An exact tie such as 3/800 = 0.00375 rounds differently if computed as one fraction, so the formula must match.)
        want = round(counts[(a, 'all_rows')] / n - counts[(b, 'all_rows')] / n, 4)
        if saved["paired_diff_point"] != want:
            raise DataError(f"paired {key}: saved difference {saved['paired_diff_point']} but the predictions give {want}")


def build(root: Path = ROOT) -> str:
    """Validate and return the exact text of data.json. Reads only; writes nothing."""
    root = Path(root)
    src_cases = read_jsonl(root / "data/test.jsonl")
    cases = {}
    for r in src_cases:
        if r["row_id"] in cases:
            raise DataError(f"duplicate case id {r['row_id']}")
        cases[r["row_id"]] = r
    preds = {m: load_predictions(root, m, cases) for m in MODELS}
    metrics_path = root / "runs/test-results/test_metrics.json"
    if not metrics_path.exists():
        raise DataError(f"missing file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    check_metrics(metrics, cases, preds)
    out_cases = []
    for rid, c in cases.items():
        case = {k: c[k] for k in CASE_KEYS}
        case["predictions"] = {m: {k: preds[m][rid][k] for k in PRED_KEYS} for m in MODELS}
        out_cases.append(case)
    sources = {"data/test.jsonl": sha256(root / "data/test.jsonl"), "runs/test-results/test_metrics.json": sha256(metrics_path)}
    for m in MODELS:
        sources[f"runs/test-results/predictions_{m}.jsonl"] = sha256(root / f"runs/test-results/predictions_{m}.jsonl")
    payload = {"cases": out_cases, "metrics": metrics,
               "meta": {"rows": len(out_cases), "groups": len({c["group_id"] for c in out_cases}), "models": list(MODELS), "sources": sources}}
    return json.dumps(payload, separators=(",", ":")) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build web/data.json from the saved test cases, predictions and metrics.")
    ap.add_argument("--check", action="store_true", help="validate and confirm the committed data.json is byte-identical; write nothing")
    ap.add_argument("--output", type=Path, help="write here instead of web/data.json")
    a = ap.parse_args(argv)
    target = a.output or (ROOT / "web/data.json")
    try:
        text = build(ROOT)
    except DataError as e:
        print(f"refusing to build dashboard data: {e}", file=sys.stderr)
        return 2
    if a.check:
        if not target.exists():
            print(f"{target} does not exist", file=sys.stderr)
            return 3
        if target.read_text() != text:
            print(f"{target} is out of date: regenerate it with python3 web/generate_data.py", file=sys.stderr)
            return 4
        print(f"{target} is byte-identical to a fresh build")
        return 0
    target.write_text(text)
    d = json.loads(text)
    print(f"Wrote {d['meta']['rows']} cases ({d['meta']['groups']} groups) and {len(MODELS)} models to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
