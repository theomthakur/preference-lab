"""Tests for web/generate_data.py. Synthetic fixtures live in a temp folder; the real data is only read."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("generate_data", REPO / "web" / "generate_data.py")
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)

N_GROUPS = 6


def make_case(g, t):
    return {"row_id": f"g{g:05d}-t{t}", "group_id": f"g{g:05d}", "split": "test", "template": t, "novel_values": g % 2 == 0,
            "rule": "in_window" if g % 2 else "too_late", "prompt": f"[L1] policy {g}\n[L2] ticket {t}",
            "label": ["ACCEPT", "DENY", "HOLD"][g % 3], "required_ids": ["L1", "L2"]}


def make_pred(case, model, correct):
    action_ok = correct or case["row_id"].endswith("t3")
    return {**{k: case[k] for k in ("row_id", "group_id", "split", "template", "novel_values", "rule", "label", "required_ids")},
            "output": json.dumps({"action": case["label"], "evidence": case["required_ids"]}), "action": case["label"], "valid": True,
            "format_ok": True, "action_ok": action_ok, "evidence_ok": correct, "correct": correct and action_ok}


def correct_fn(model, case):
    return model != "base" and (case["row_id"].endswith("t0") or int(case["group_id"][1:]) % 2 == 0)


def metrics_for(preds, cases):
    held = {c["row_id"] for c in cases if c["template"] == 3}
    ids = {"all_rows": {c["row_id"] for c in cases}, "seen_template": {c["row_id"] for c in cases} - held, "held_out_template": held}
    models, count = {}, {}
    for m, ps in preds.items():
        models[m] = {}
        for view, s in ids.items():
            k = sum(ps[r]["correct"] for r in s); count[(m, view)] = k
            models[m][view] = {"n_rows": len(s), "point": round(k / len(s), 4), "ci95": [0.0, 1.0]}
    n = len(cases)
    paired = {key: {"n_rows": n, "point": 0, "ci95": [0, 0], "paired_diff_point": round(count[(a, "all_rows")] / n - count[(b, "all_rows")] / n, 4),
                    "paired_diff_ci95": [0, 0]} for key, (a, b) in gd.PAIRS.items()}
    return {"models": models, "paired": paired, "predicted_actions": {}}


def write_root(root, mutate=None):
    cases = [make_case(g, t) for g in range(N_GROUPS) for t in (0, 3)]
    preds = {m: {c["row_id"]: make_pred(c, m, correct_fn(m, c)) for c in cases} for m in gd.MODELS}
    (root / "data").mkdir(parents=True, exist_ok=True); (root / "runs/test-results").mkdir(parents=True, exist_ok=True); (root / "web").mkdir(exist_ok=True)
    metrics = metrics_for(preds, cases)
    if mutate:
        mutate(cases, preds, metrics)
    (root / "data/test.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    for m in gd.MODELS:
        (root / f"runs/test-results/predictions_{m}.jsonl").write_text("\n".join(json.dumps(p) for p in (preds[m].values() if isinstance(preds[m], dict) else preds[m])) + "\n")
    (root / "runs/test-results/test_metrics.json").write_text(json.dumps(metrics))
    return root


def test_valid_synthetic_root_builds(tmp_path):
    root = write_root(tmp_path)
    d = json.loads(gd.build(root))
    assert d["meta"]["rows"] == 12 and d["meta"]["groups"] == 6 and len(d["cases"]) == 12
    assert set(d["cases"][0]["predictions"]) == set(gd.MODELS)
    assert set(d["cases"][0]["predictions"]["sft"]) == set(gd.PRED_KEYS)
    assert len(d["meta"]["sources"]) == 2 + len(gd.MODELS)


def test_committed_data_json_is_byte_identical_to_a_fresh_build():
    """The committed dashboard data must be exactly what the generator produces from the committed sources."""
    assert (REPO / "web/data.json").read_text() == gd.build(REPO)


def test_build_never_writes_into_the_sources(tmp_path):
    root = write_root(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    gd.build(root)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("field,value", [("group_id", "g00099"), ("template", 1), ("label", "HOLD"), ("rule", "other_rule"),
                                         ("required_ids", ["L9"]), ("novel_values", "maybe"), ("split", "val")])
def test_misaligned_prediction_fields_are_rejected(tmp_path, field, value):
    def mutate(cases, preds, metrics):
        row = preds["sft"][cases[3]["row_id"]]
        row[field] = value if field != "label" else ("DENY" if cases[3]["label"] != "DENY" else "ACCEPT")
    with pytest.raises(gd.DataError, match="misaligned|split"):
        gd.build(write_root(tmp_path, mutate))


def test_swapped_group_ids_between_two_rows_are_rejected(tmp_path):
    """The exact tampering that the previous generator let through: two rows swap group ids, ids and accuracy stay the same."""
    def mutate(cases, preds, metrics):
        a, b = preds["grpo"][cases[0]["row_id"]], preds["grpo"][cases[2]["row_id"]]
        a["group_id"], b["group_id"] = b["group_id"], a["group_id"]
    with pytest.raises(gd.DataError, match="misaligned on 'group_id'"):
        gd.build(write_root(tmp_path, mutate))


def test_missing_extra_and_duplicate_predictions_are_rejected(tmp_path):
    def drop(cases, preds, metrics):
        del preds["dpo_clean"][cases[5]["row_id"]]
    with pytest.raises(gd.DataError, match="no prediction"):
        gd.build(write_root(tmp_path / "a", drop))
    def extra(cases, preds, metrics):
        preds["base"]["g99999-t0"] = {**preds["base"][cases[0]["row_id"]], "row_id": "g99999-t0"}
    with pytest.raises(gd.DataError, match="match no case"):
        gd.build(write_root(tmp_path / "b", extra))
    def dup(cases, preds, metrics):
        preds["sft"] = list(preds["sft"].values()) + [preds["sft"][cases[1]["row_id"]]]
    with pytest.raises(gd.DataError, match="duplicate prediction"):
        gd.build(write_root(tmp_path / "c", dup))


def test_inconsistent_or_non_boolean_flags_are_rejected(tmp_path):
    def bad_flags(cases, preds, metrics):
        p = preds["sft"][cases[0]["row_id"]]; p["correct"] = True; p["evidence_ok"] = False; p["action_ok"] = True
    with pytest.raises(gd.DataError, match="inconsistent flags"):
        gd.build(write_root(tmp_path / "a", bad_flags))
    def non_bool(cases, preds, metrics):
        preds["sft"][cases[0]["row_id"]]["correct"] = "yes"
    with pytest.raises(gd.DataError, match="non-boolean"):
        gd.build(write_root(tmp_path / "b", non_bool))


def test_saved_metrics_must_match_the_predictions_exactly(tmp_path):
    def wrong_point(cases, preds, metrics):
        metrics["models"]["sft"]["held_out_template"]["point"] += 0.0001
    with pytest.raises(gd.DataError, match="saved accuracy"):
        gd.build(write_root(tmp_path / "a", wrong_point))
    def wrong_n(cases, preds, metrics):
        metrics["models"]["grpo"]["all_rows"]["n_rows"] = 13
    with pytest.raises(gd.DataError, match="n_rows"):
        gd.build(write_root(tmp_path / "b", wrong_n))
    def wrong_pair(cases, preds, metrics):
        metrics["paired"]["grpo_vs_sft"]["paired_diff_point"] += 0.0001
    with pytest.raises(gd.DataError, match="paired grpo_vs_sft"):
        gd.build(write_root(tmp_path / "c", wrong_pair))
    def missing_pair(cases, preds, metrics):
        del metrics["paired"]["sft_vs_base"]
    with pytest.raises(gd.DataError, match="no paired entry"):
        gd.build(write_root(tmp_path / "d", missing_pair))


def test_paired_difference_uses_the_same_arithmetic_as_the_analysis_tool():
    """An exact tie: 3/800 = 0.00375. Rounding the fraction gives 0.0037, subtracting the unrounded accuracies first gives 0.0038.
    The saved test_metrics.json holds 0.0038 (analysis tool), so the validator must use the second form."""
    assert round(3 / 800, 4) == 0.0037 and round(79 / 800 - 76 / 800, 4) == 0.0038
    saved = json.loads((REPO / "runs/test-results/test_metrics.json").read_text())["paired"]["dpo_corrupt_vs_dpo_clean"]["paired_diff_point"]
    assert saved == 0.0038


def test_cli_modes(tmp_path, monkeypatch, capsys):
    root = write_root(tmp_path)
    monkeypatch.setattr(gd, "ROOT", root)
    target = root / "web/data.json"
    assert gd.main(["--check"]) == 3                                   # nothing written yet
    assert gd.main([]) == 0 and target.exists()                        # default writes web/data.json
    assert gd.main(["--check"]) == 0                                   # byte-identical
    target.write_text(target.read_text().replace('"rows":12', '"rows":13'))
    assert gd.main(["--check"]) == 4                                   # a stale or edited file is detected
    other = tmp_path / "elsewhere.json"
    assert gd.main(["--output", str(other)]) == 0 and other.exists()
    assert target.read_text().count('"rows":13') == 1                  # --output did not touch web/data.json


def test_cli_refuses_to_write_when_sources_do_not_line_up(tmp_path, monkeypatch, capsys):
    def mutate(cases, preds, metrics):
        preds["sft"][cases[0]["row_id"]]["template"] = 1
    root = write_root(tmp_path, mutate)
    monkeypatch.setattr(gd, "ROOT", root)
    assert gd.main([]) == 2
    assert not (root / "web/data.json").exists()
    assert "refusing to build" in capsys.readouterr().err
