"""Analysis of the held-out test, computed ONLY from the saved predictions in runs/test-results (no model is run here).

  python -m preference_lab.test_analysis
"""
from __future__ import annotations
import json, random
from pathlib import Path

OUT = Path("runs/test-results")
MODELS = ["base", "sft", "dpo_clean", "dpo_corrupt", "grpo"]


def load_preds(name):
    return {json.loads(l)["row_id"]: json.loads(l) for l in (OUT / f"predictions_{name}.jsonl").read_text().splitlines() if l.strip()}


def boot(correct_a: dict, correct_b: dict | None, groups: dict, n=2000, seed=7):
    """Group bootstrap over the rows in correct_a (optionally paired with correct_b). groups: group_id -> row ids in this subset."""
    groups = {g: rs for g, rs in groups.items() if rs}
    r = random.Random(seed); gids = sorted(groups)
    point_a = sum(correct_a[x] for g in gids for x in groups[g]) / sum(len(groups[g]) for g in gids)
    accs, diffs = [], []
    for _ in range(n):
        pick = [r.choice(gids) for _ in gids]
        rs = [x for g in pick for x in groups[g]]
        accs.append(sum(correct_a[x] for x in rs) / len(rs))
        if correct_b is not None:
            diffs.append(sum(correct_a[x] - correct_b[x] for x in rs) / len(rs))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    res = {"n_rows": sum(len(v) for v in groups.values()), "point": round(point_a, 4), "ci95": [round(q(accs, .025), 4), round(q(accs, .975), 4)]}
    if correct_b is not None:
        pb = sum(correct_b[x] for g in gids for x in groups[g]) / sum(len(groups[g]) for g in gids)
        res["paired_diff_point"] = round(point_a - pb, 4)
        res["paired_diff_ci95"] = [round(q(diffs, .025), 4), round(q(diffs, .975), 4)]
    return res


def main():
    P = {m: load_preds(m) for m in MODELS}
    correct = {m: {rid: int(p["correct"]) for rid, p in P[m].items()} for m in MODELS}
    ref_rows = P["sft"]
    all_groups, seen_groups, held_groups = {}, {}, {}
    for rid, p in ref_rows.items():
        all_groups.setdefault(p["group_id"], []).append(rid)
        (held_groups if p["template"] == 3 else seen_groups).setdefault(p["group_id"], []).append(rid)
    models = {}
    for m in MODELS:
        models[m] = {"all_rows": boot(correct[m], None, all_groups),
                     "seen_template": boot(correct[m], None, seen_groups),
                     "held_out_template": boot(correct[m], None, held_groups)}
    pairs = {"sft_vs_base": ("sft", "base"), "dpo_clean_vs_sft": ("dpo_clean", "sft"),
             "dpo_corrupt_vs_dpo_clean": ("dpo_corrupt", "dpo_clean"), "grpo_vs_sft": ("grpo", "sft")}
    paired = {k: boot(correct[a], correct[b], all_groups) for k, (a, b) in pairs.items()}
    actions = {m: {} for m in MODELS}
    for m in MODELS:
        for p in P[m].values():
            actions[m][str(p["action"])] = actions[m].get(str(p["action"]), 0) + 1
    out = {"models": models, "paired": paired, "predicted_actions": actions}
    (OUT / "test_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({m: {k: (v["point"], v["ci95"]) for k, v in models[m].items()} for m in MODELS}, indent=1))
    print(json.dumps({k: (v["paired_diff_point"], v["paired_diff_ci95"]) for k, v in paired.items()}, indent=1))
    print("predicted actions:", json.dumps(actions))


if __name__ == "__main__":
    main()
