"""Error breakdown for saved predictions, so every experiment is diagnosed the same way.

  python -m preference_lab.errors runs/sft-val/predictions.jsonl --split val

Separates formatting, wrong action and wrong evidence, says HOW the evidence is wrong (missed lines, extra lines, a note line, the
superseded line, another real fact line), and shows accuracy by the number of evidence lines and by rule. Reads saved files only.
Use it on validation predictions. The held-out test is not used to choose anything.
"""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path
from .oracle import parse_answer
from .splits import load


def line_kinds(prompt: str) -> dict[str, str]:
    kinds = {}
    for line in prompt.splitlines():
        m = re.match(r"\[(L\d+)\]\s*(.*)", line)
        if m:
            kinds[m.group(1)] = "note" if m.group(2).startswith("Note:") else ("superseded" if "superseded" in m.group(2) else "fact")
    return kinds


def breakdown(preds: list[dict], cases: dict[str, dict]) -> dict:
    """preds: saved prediction rows. cases: row_id -> source case (with prompt and required_ids)."""
    top, detail = collections.Counter(), collections.Counter()
    by_size = collections.defaultdict(lambda: [0, 0, 0]); by_rule = collections.defaultdict(lambda: [0, 0, 0]); overlap = []
    for r in preds:
        c = cases[r["row_id"]]; req = set(c["required_ids"]); kinds = line_kinds(c["prompt"])
        for key, table in ((len(req), by_size), (c["rule"], by_rule)):
            table[key][0] += 1; table[key][1] += bool(r["action_ok"]); table[key][2] += bool(r["correct"])
        a = parse_answer(r["output"])
        if a is None or a["evidence"] is None:
            top["no valid evidence list"] += 1; continue
        s = set(a["evidence"]); overlap.append(len(s & req) / max(1, len(s | req)))
        if r["correct"]: top["strict correct"] += 1; continue
        if not r["action_ok"]: top["wrong action"] += 1
        if r["action_ok"]:
            top["right action, wrong evidence"] += 1
            detail["missed some required lines only" if s < req else "all required plus extra lines" if s > req else "some right, some wrong" if s & req else "none of the required lines"] += 1
            for i in s - req:
                detail[{"note": "cited a note line", "superseded": "cited the superseded line", "fact": "cited another real fact line"}.get(kinds.get(i), "cited an id not in the prompt")] += 1
    n = len(preds)
    return {"n": n, "top": dict(top), "evidence_detail": dict(detail), "mean_evidence_overlap": round(sum(overlap) / max(1, len(overlap)), 3),
            "by_required_lines": {k: {"n": t[0], "action": round(t[1] / t[0], 3), "strict": round(t[2] / t[0], 3)} for k, t in sorted(by_size.items())},
            "by_rule": {k: {"n": t[0], "action": round(t[1] / t[0], 3), "strict": round(t[2] / t[0], 3)} for k, t in sorted(by_rule.items(), key=lambda x: -x[1][0])}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("predictions", type=Path); ap.add_argument("--split", default="val")
    a = ap.parse_args()
    cases = {r["row_id"]: r for r in load(a.split)}
    preds = [json.loads(l) for l in a.predictions.read_text().splitlines() if l.strip()]
    print(json.dumps(breakdown(preds, cases), indent=1))


if __name__ == "__main__":
    main()
