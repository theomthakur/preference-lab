"""Preference pairs (chosen vs rejected) built only from the 'pool' split, with the rule oracle deciding which is better.

  python -m preference_lab.pairs build       # writes data/pairs_clean.jsonl, data/pairs_corrupt.jsonl, data/corruption_flags.json

chosen   = the oracle's exact answer.
rejected = a hard negative that is still valid JSON, so formatting alone cannot separate them:
  wrong_action        - another action, same evidence ids
  missing_evidence    - right action, one required id dropped (or a note id added when only one id is required)
  extra_evidence      - right action, one extra id added (can be a note or superseded line)
  fabricated_evidence - right action, one required id replaced by an id that is not in the prompt
  wrong_both          - another action with unrelated evidence ids of the same count
Every pair is checked with the verifier: chosen must score correct, rejected must not.
The corrupt file is the clean file with the chosen/rejected order swapped for a seeded 20% of groups. The learner never sees the
flags; they live in corruption_flags.json.
"""
from __future__ import annotations
import json, random, re, sys
from pathlib import Path
from .data import answer_json
from .oracle import ACTIONS, verify
from .splits import load

TYPES = ["wrong_action", "missing_evidence", "extra_evidence", "fabricated_evidence", "wrong_both"]


def _ids(prompt: str) -> list[str]:
    return re.findall(r"^\[(L\d+)\]", prompt, re.M)


def make_rejected(row: dict, r: random.Random) -> tuple[str, str]:
    lab, req, all_ids = row["label"], list(row["required_ids"]), _ids(row["prompt"])
    other = [a for a in ACTIONS if a != lab]
    others = [i for i in all_ids if i not in req]
    kind = r.choice(TYPES)
    if kind == "wrong_action":
        a, ev = r.choice(other), req
    elif kind == "missing_evidence":
        if len(req) > 1:
            ev = r.sample(req, len(req) - 1)
        else:
            ev = req + [r.choice(others)]
        a = lab
    elif kind == "extra_evidence":
        a, ev = lab, req + [r.choice(others)]
    elif kind == "fabricated_evidence":
        a = lab
        ev = list(req); ev[r.randrange(len(ev))] = f"L{len(all_ids) + r.randint(1, 9)}"
    else:
        a = r.choice(other); ev = r.sample(others, min(len(req), len(others)))
    return kind, answer_json(a, ev)


def build_pairs(seed: int = 11) -> tuple[list[dict], list[dict], dict]:
    r = random.Random(seed)
    clean = []
    for row in load("pool"):
        chosen = answer_json(row["label"], row["required_ids"])
        kind, rejected = make_rejected(row, r)
        assert chosen != rejected
        assert verify(chosen, row["label"], row["required_ids"])["correct"]
        assert not verify(rejected, row["label"], row["required_ids"])["correct"], (row["row_id"], kind, rejected)
        clean.append({"row_id": row["row_id"], "group_id": row["group_id"], "system": row["system"], "prompt": row["prompt"],
                      "chosen": chosen, "rejected": rejected, "rejected_type": kind})
    flip = set(r.sample(range(len(clean)), len(clean) // 5))
    corrupt = []
    for i, p in enumerate(clean):
        q = dict(p)
        if i in flip:
            q["chosen"], q["rejected"] = p["rejected"], p["chosen"]
        corrupt.append(q)
    flags = {"seed": seed, "corrupted_row_ids": sorted(clean[i]["row_id"] for i in flip), "fraction": len(flip) / len(clean)}
    return clean, corrupt, flags


def main():
    clean, corrupt, flags = build_pairs()
    out = Path("data")
    for name, rows in (("pairs_clean", clean), ("pairs_corrupt", corrupt)):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True) for x in rows) + "\n")
    (out / "corruption_flags.json").write_text(json.dumps(flags, indent=1))
    import collections, statistics
    print("pairs:", len(clean), "types:", dict(collections.Counter(p["rejected_type"] for p in clean)))
    d = [len(p["rejected"]) - len(p["chosen"]) for p in clean]
    print(f"rejected minus chosen length (chars): mean {statistics.mean(d):.2f}, sd {statistics.pstdev(d):.2f}; corrupted fraction {flags['fraction']:.2f}")


if __name__ == "__main__":
    main()
