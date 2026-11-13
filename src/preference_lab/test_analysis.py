"""Analysis of the held-out test, computed ONLY from saved predictions. No model is loaded and nothing is generated.

  python -m preference_lab.test_analysis                         # recompute and PRINT; writes nothing
  python -m preference_lab.test_analysis --output new.json       # also write to a NEW file (refuses to overwrite)
  python -m preference_lab.test_analysis --output x.json --overwrite
  python -m preference_lab.test_analysis --check-against runs/test-results/test_metrics.json   # compare, change nothing

Safety rules (see tests/test_test_analysis.py):
  * The prediction files are validated against data/test.jsonl before any number is computed: every expected row must be
    present exactly once, no unexpected rows, and group_id / template must match the test file. Missing, duplicated or
    misaligned rows raise PredictionError instead of producing a number.
  * Existing files are never overwritten unless --overwrite is passed. The default is to write nothing.
  * The group bootstrap resamples whole groups (a group has two rendered rows that are not independent).

Statistical limits, in short: a percentile bootstrap of an all-zero result is always [0, 0] (every resample has zero
successes), so it carries no uncertainty information. For zero-success cells this module reports a supplementary exact
(Clopper-Pearson) upper bound as plain arithmetic. It is not a new experimental result and does not replace the original.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path

DEFAULT_DIR = Path("runs/test-results")
DEFAULT_TEST = Path("data/test.jsonl")
MODELS = ["base", "sft", "dpo_clean", "dpo_corrupt", "grpo"]
PAIRS = {"sft_vs_base": ("sft", "base"), "dpo_clean_vs_sft": ("dpo_clean", "sft"),
         "dpo_corrupt_vs_dpo_clean": ("dpo_corrupt", "dpo_clean"), "grpo_vs_sft": ("grpo", "sft")}
N_RESAMPLES = 2000
SEED = 7


class PredictionError(ValueError):
    """The saved predictions do not line up with the test file."""


def load_expected(test_file: Path = DEFAULT_TEST) -> dict[str, tuple[str, int]]:
    """row_id -> (group_id, template) from the frozen test file."""
    rows = [json.loads(l) for l in Path(test_file).read_text().splitlines() if l.strip()]
    ids = [r["row_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise PredictionError("test file has duplicate row_ids")
    return {r["row_id"]: (r["group_id"], r["template"]) for r in rows}


def load_predictions(path: Path, model: str, expected: dict) -> dict[str, dict]:
    """Read one model's predictions and check them against the expected rows. Returns row_id -> prediction."""
    if not Path(path).exists():
        raise PredictionError(f"{model}: prediction file missing: {path}")
    preds, dupes = {}, []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except Exception as e:
            raise PredictionError(f"{model}: line {n} is not valid JSON ({e})")
        for key in ("row_id", "group_id", "template", "correct"):
            if key not in p:
                raise PredictionError(f"{model}: line {n} lacks '{key}'")
        if p["correct"] not in (True, False, 0, 1) or isinstance(p["correct"], str):
            raise PredictionError(f"{model}: line {n} has a non-boolean 'correct': {p['correct']!r}")
        if p["row_id"] in preds:
            dupes.append(p["row_id"])
        preds[p["row_id"]] = p
    if dupes:
        raise PredictionError(f"{model}: duplicated rows, for example {dupes[:3]} ({len(dupes)} in total)")
    missing = sorted(set(expected) - set(preds)); extra = sorted(set(preds) - set(expected))
    if missing:
        raise PredictionError(f"{model}: {len(missing)} missing rows, for example {missing[:3]}")
    if extra:
        raise PredictionError(f"{model}: {len(extra)} unexpected rows, for example {extra[:3]}")
    for rid, p in preds.items():
        g, t = expected[rid]
        if p["group_id"] != g or p["template"] != t:
            raise PredictionError(f"{model}: row {rid} is misaligned (file says group {p['group_id']}, template {p['template']}; "
                                  f"test file says group {g}, template {t})")
    return preds


def zero_success_upper_95(n: int) -> float:
    """Two-sided 95% exact (Clopper-Pearson) upper bound when 0 of n trials succeed: 1 - 0.025**(1/n).
    Treats the n units as independent, so it is optimistic when units are clustered (here: two rows per group)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - math.exp(math.log(0.025) / n)


def boot(correct_a: dict, correct_b: dict | None, groups: dict, n: int = N_RESAMPLES, seed: int = SEED, return_samples: bool = False):
    """Group-level percentile bootstrap. groups: group_id -> row ids in this subset. Whole groups are resampled together.
    correct_a / correct_b: row_id -> 0/1. If correct_b is given, also returns the paired difference (a minus b)."""
    groups = {g: list(rs) for g, rs in groups.items() if rs}
    if not groups:
        raise PredictionError("no rows to analyse")
    for g, rs in groups.items():
        for x in rs:
            if x not in correct_a or (correct_b is not None and x not in correct_b):
                raise PredictionError(f"row {x} in group {g} has no prediction")
    r = random.Random(seed); gids = sorted(groups)
    total = sum(len(groups[g]) for g in gids)
    point_a = sum(correct_a[x] for g in gids for x in groups[g]) / total
    accs, diffs = [], []
    for _ in range(n):
        pick = [r.choice(gids) for _ in gids]
        rs = [x for g in pick for x in groups[g]]
        accs.append(sum(correct_a[x] for x in rs) / len(rs))
        if correct_b is not None:
            diffs.append(sum(correct_a[x] - correct_b[x] for x in rs) / len(rs))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    res = {"n_rows": total, "point": round(point_a, 4), "ci95": [round(q(accs, .025), 4), round(q(accs, .975), 4)]}
    if correct_b is not None:
        pb = sum(correct_b[x] for g in gids for x in groups[g]) / total
        res["paired_diff_point"] = round(point_a - pb, 4)
        res["paired_diff_ci95"] = [round(q(diffs, .025), 4), round(q(diffs, .975), 4)]
    if return_samples:
        res["_samples"] = accs
    return res


def analyze(results_dir: Path = DEFAULT_DIR, test_file: Path = DEFAULT_TEST, models: list[str] = MODELS) -> dict:
    expected = load_expected(test_file)
    P = {m: load_predictions(Path(results_dir) / f"predictions_{m}.jsonl", m, expected) for m in models}
    correct = {m: {rid: int(bool(p["correct"])) for rid, p in P[m].items()} for m in models}
    all_g, seen_g, held_g = {}, {}, {}
    for rid, (g, t) in expected.items():
        all_g.setdefault(g, []).append(rid)
        (held_g if t == 3 else seen_g).setdefault(g, []).append(rid)
    out_models, supplementary = {}, {}
    for m in models:
        out_models[m] = {"all_rows": boot(correct[m], None, all_g), "seen_template": boot(correct[m], None, seen_g),
                         "held_out_template": boot(correct[m], None, held_g)}
        for cell, v in out_models[m].items():
            if v["point"] == 0.0:
                supplementary[f"{m}/{cell}"] = {
                    "note": "zero successes: the bootstrap interval [0, 0] carries no uncertainty; exact upper bounds below are plain arithmetic",
                    "exact_95_upper_if_rows_independent": round(zero_success_upper_95(v["n_rows"]), 5),
                    "exact_95_upper_with_groups_as_units": round(zero_success_upper_95(len({expected[r][0] for r in expected})), 5)}
    paired = {k: boot(correct[a], correct[b], all_g) for k, (a, b) in PAIRS.items() if a in correct and b in correct}
    actions = {m: {} for m in models}
    for m in models:
        for p in P[m].values():
            actions[m][str(p.get("action"))] = actions[m].get(str(p.get("action")), 0) + 1
    return {"models": out_models, "paired": paired, "predicted_actions": actions, "supplementary": supplementary,
            "method": {"bootstrap": "group-level percentile", "resamples": N_RESAMPLES, "seed": SEED, "source": "saved predictions only"}}


def compare(new: dict, saved_path: Path) -> list[str]:
    """Differences between a fresh recomputation and a saved metrics file, for models and paired comparisons only."""
    saved = json.loads(Path(saved_path).read_text()); diffs = []
    for section in ("models", "paired"):
        for k, v in new[section].items():
            if k not in saved.get(section, {}):
                diffs.append(f"{section}/{k} missing from saved file"); continue
            if v != saved[section][k]:
                diffs.append(f"{section}/{k}: recomputed {v} != saved {saved[section][k]}")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_DIR); ap.add_argument("--test-file", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--output", type=Path, help="write the recomputed metrics here (a new file)")
    ap.add_argument("--overwrite", action="store_true", help="allow --output to replace an existing file")
    ap.add_argument("--check-against", type=Path, help="compare with a saved metrics file and report differences; changes nothing")
    a = ap.parse_args(argv)
    if a.output and a.output.exists() and not a.overwrite:
        print(f"refusing to overwrite {a.output}; choose a new path or pass --overwrite", file=sys.stderr); return 2
    try:
        res = analyze(a.results_dir, a.test_file)
    except PredictionError as e:
        print(f"prediction check failed: {e}", file=sys.stderr); return 3
    print(json.dumps({m: {k: (v["point"], v["ci95"]) for k, v in res["models"][m].items()} for m in res["models"]}, indent=1))
    print(json.dumps({k: (v["paired_diff_point"], v["paired_diff_ci95"]) for k, v in res["paired"].items()}, indent=1))
    for k, v in res["supplementary"].items():
        print(f"zero-success cell {k}: exact 95% upper bound {v['exact_95_upper_if_rows_independent']} (rows independent), "
              f"{v['exact_95_upper_with_groups_as_units']} (groups as units)")
    if a.check_against:
        d = compare(res, a.check_against)
        print("check-against:", "identical to the saved file" if not d else f"{len(d)} differences")
        for x in d[:10]: print("  ", x)
        if d: return 4
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(res, indent=2))
        print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
