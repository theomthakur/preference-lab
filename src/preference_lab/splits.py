"""Build the dataset to disk with manifests, and audit it for leakage and balance.

  python -m preference_lab.splits build    # writes data/{sft,pool,val,test}.jsonl and data/manifest.json
  python -m preference_lab.splits audit    # re-checks leakage and balance from the files on disk
"""
from __future__ import annotations
import collections, hashlib, json, sys
from pathlib import Path
from .data import build_dataset, GENERATOR_VERSION, SEEN_TEMPLATES, HELD_OUT_TEMPLATE, TRAIN_WINDOWS, NOVEL_WINDOWS

DATA = Path("data")
SEED = 7


def build(seed: int = SEED, out: Path = DATA) -> dict:
    rows = build_dataset(seed)
    out.mkdir(exist_ok=True)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    files = {}
    for split, rs in by.items():
        p = out / f"{split}.jsonl"
        p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rs) + "\n")
        files[split] = {"rows": len(rs), "groups": len({r["group_id"] for r in rs}), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    manifest = {"generator_version": GENERATOR_VERSION, "seed": seed, "label_origin": "synthetic rule oracle, not human judgments", "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load(split: str, out: Path = DATA) -> list[dict]:
    return [json.loads(l) for l in (out / f"{split}.jsonl").read_text().splitlines() if l.strip()]


def audit(out: Path = DATA) -> list[str]:
    """Return a list of problems (empty = clean)."""
    problems = []
    splits = {s: load(s, out) for s in ("sft", "pool", "val", "test")}
    groups = {s: {r["group_id"] for r in rs} for s, rs in splits.items()}
    hashes = {s: {r["group_hash"] for r in rs} for s, rs in splits.items()}
    prompts = {s: {r["prompt"] for r in rs} for s, rs in splits.items()}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if groups[a] & groups[b]: problems.append(f"group ids shared between {a} and {b}")
            if hashes[a] & hashes[b]: problems.append(f"identical underlying cases shared between {a} and {b}")
            if prompts[a] & prompts[b]: problems.append(f"identical prompts shared between {a} and {b}")
    for s, rs in splits.items():
        c = collections.Counter(r["label"] for r in rs if r["template"] in SEEN_TEMPLATES)
        n = sum(c.values())
        for lab in ("ACCEPT", "DENY", "HOLD"):
            if not (0.30 <= c[lab] / n <= 0.37): problems.append(f"{s}: label {lab} share {c[lab]/n:.2f} outside 0.30-0.37")
    for s in ("sft", "pool", "val"):
        if any(r["template"] == HELD_OUT_TEMPLATE for r in splits[s]): problems.append(f"{s} contains the held-out template")
        if any(r["policy"]["window"] in NOVEL_WINDOWS for r in splits[s]): problems.append(f"{s} contains a test-only window value")
    test = splits["test"]
    if not any(r["policy"]["window"] in NOVEL_WINDOWS for r in test): problems.append("test has no novel window values")
    if not any(r["template"] == HELD_OUT_TEMPLATE for r in test): problems.append("test has no held-out template rows")
    for s, rs in splits.items():
        for r in rs:
            if len(r["required_ids"]) == 0 or any(i not in r["prompt"] for i in r["required_ids"]): problems.append(f"{r['row_id']}: bad required ids"); break
    return problems


def report(out: Path = DATA) -> str:
    lines = []
    for s in ("sft", "pool", "val", "test"):
        rs = load(s, out)
        c = collections.Counter(r["label"] for r in rs)
        lines.append(f"{s}: rows={len(rs)} groups={len({r['group_id'] for r in rs})} labels={dict(c)}")
    return "\n".join(lines)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if cmd == "build":
        m = build(); print(json.dumps(m, indent=2))
    problems = audit()
    print(report())
    print("AUDIT:", "clean" if not problems else problems)
    sys.exit(1 if problems else 0)
