"""Tests for the saved-prediction analysis. They use small synthetic files in a temp folder and never touch runs/test-results."""
import json
import pytest
from preference_lab.test_analysis import (MODELS, PredictionError, analyze, boot, compare, load_expected, load_predictions, main,
                                          zero_success_upper_95)

N_GROUPS = 6


def write_test_file(path, n_groups=N_GROUPS):
    rows = []
    for g in range(n_groups):
        for t in (0, 3):
            rows.append({"row_id": f"g{g:05d}-t{t}", "group_id": f"g{g:05d}", "template": t})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def write_preds(dirpath, rows, correct_fn, models=MODELS):
    dirpath.mkdir(parents=True, exist_ok=True)
    for m in models:
        lines = [json.dumps({"row_id": r["row_id"], "group_id": r["group_id"], "template": r["template"],
                             "correct": bool(correct_fn(m, r)), "action": "DENY"}) for r in rows]
        (dirpath / f"predictions_{m}.jsonl").write_text("\n".join(lines) + "\n")


def setup(tmp_path, correct_fn=lambda m, r: m != "base"):
    test = tmp_path / "test.jsonl"; rows = write_test_file(test)
    res = tmp_path / "results"; write_preds(res, rows, correct_fn)
    return test, res, rows


def test_valid_data_analyses(tmp_path):
    test, res, rows = setup(tmp_path)
    out = analyze(res, test)
    assert out["models"]["sft"]["all_rows"]["point"] == 1.0
    assert out["models"]["base"]["all_rows"]["point"] == 0.0
    assert out["paired"]["sft_vs_base"]["paired_diff_point"] == 1.0


def test_missing_rows_are_rejected(tmp_path):
    test, res, rows = setup(tmp_path)
    p = res / "predictions_sft.jsonl"
    p.write_text("\n".join(p.read_text().splitlines()[:-2]) + "\n")
    with pytest.raises(PredictionError, match="missing"):
        analyze(res, test)


def test_duplicated_rows_are_rejected(tmp_path):
    test, res, rows = setup(tmp_path)
    p = res / "predictions_dpo_clean.jsonl"; lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(PredictionError, match="duplicated"):
        analyze(res, test)


def test_unexpected_rows_are_rejected(tmp_path):
    test, res, rows = setup(tmp_path)
    p = res / "predictions_grpo.jsonl"
    extra = json.dumps({"row_id": "g99999-t0", "group_id": "g99999", "template": 0, "correct": True})
    p.write_text(p.read_text() + extra + "\n")
    with pytest.raises(PredictionError, match="unexpected"):
        analyze(res, test)


@pytest.mark.parametrize("field,value", [("group_id", "g00003"), ("template", 1)])
def test_misaligned_rows_are_rejected(tmp_path, field, value):
    test, res, rows = setup(tmp_path)
    p = res / "predictions_sft.jsonl"; lines = [json.loads(l) for l in p.read_text().splitlines()]
    lines[0][field] = value
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    with pytest.raises(PredictionError, match="misaligned"):
        analyze(res, test)


def test_missing_model_file_and_bad_values_are_rejected(tmp_path):
    test, res, rows = setup(tmp_path)
    (res / "predictions_base.jsonl").unlink()
    with pytest.raises(PredictionError, match="missing"):
        analyze(res, test)
    (tmp_path / "b").mkdir()
    test2, res2, _ = setup(tmp_path / "b")
    p = res2 / "predictions_sft.jsonl"; lines = [json.loads(l) for l in p.read_text().splitlines()]
    lines[0]["correct"] = "yes"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    with pytest.raises(PredictionError, match="non-boolean"):
        analyze(res2, test2)
    (res2 / "predictions_sft.jsonl").write_text("not json\n")
    with pytest.raises(PredictionError, match="not valid JSON"):
        analyze(res2, test2)


def test_bootstrap_resamples_whole_groups(tmp_path):
    """Both rows of a group always share an outcome here. If groups are resampled whole, every resampled accuracy is a multiple of
    1/6 (six groups). Row-level resampling of 12 rows would also produce odd multiples of 1/12."""
    groups = {f"g{i}": [f"g{i}-t0", f"g{i}-t3"] for i in range(6)}
    correct = {r: int(i % 2 == 0) for i, (g, rs) in enumerate(groups.items()) for r in rs}
    res = boot(correct, None, groups, n=300, seed=3, return_samples=True)
    assert all(abs(s * 6 - round(s * 6)) < 1e-9 for s in res["_samples"])
    assert len({round(s, 6) for s in res["_samples"]}) > 3


def test_bootstrap_is_deterministic_and_paired(tmp_path):
    groups = {f"g{i}": [f"g{i}-t0", f"g{i}-t3"] for i in range(8)}
    a = {r: 1 for rs in groups.values() for r in rs}; b = {r: 0 for rs in groups.values() for r in rs}
    r1, r2 = boot(a, b, groups, seed=5), boot(a, b, groups, seed=5)
    assert r1 == r2 and r1["paired_diff_point"] == 1.0 and r1["paired_diff_ci95"] == [1.0, 1.0]


def test_zero_successes_give_a_degenerate_interval_and_an_exact_bound(tmp_path):
    test, res, rows = setup(tmp_path)
    out = analyze(res, test)
    assert out["models"]["base"]["all_rows"]["ci95"] == [0.0, 0.0]            # bootstrap cannot show uncertainty here
    cell = out["supplementary"]["base/all_rows"]
    assert cell["exact_95_upper_if_rows_independent"] > 0                       # the exact bound can
    assert abs(zero_success_upper_95(800) - 0.0046) < 1e-4
    assert abs(zero_success_upper_95(400) - 0.00918) < 1e-4
    assert zero_success_upper_95(12) > zero_success_upper_95(800)


def test_default_run_writes_nothing_and_refuses_to_overwrite(tmp_path, capsys):
    test, res, rows = setup(tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert main(["--results-dir", str(res), "--test-file", str(test)]) == 0
    assert sorted(p.name for p in tmp_path.rglob("*")) == before                # nothing was written
    existing = tmp_path / "metrics.json"; existing.write_text("ORIGINAL")
    assert main(["--results-dir", str(res), "--test-file", str(test), "--output", str(existing)]) == 2
    assert existing.read_text() == "ORIGINAL"                                   # not overwritten
    assert main(["--results-dir", str(res), "--test-file", str(test), "--output", str(existing), "--overwrite"]) == 0
    assert json.loads(existing.read_text())["models"]["sft"]


def test_bad_predictions_give_a_nonzero_exit_and_no_output_file(tmp_path, capsys):
    test, res, rows = setup(tmp_path)
    p = res / "predictions_sft.jsonl"; p.write_text(p.read_text().splitlines()[0] + "\n")
    out = tmp_path / "new.json"
    assert main(["--results-dir", str(res), "--test-file", str(test), "--output", str(out)]) == 3
    assert not out.exists()


def test_check_against_detects_differences(tmp_path):
    test, res, rows = setup(tmp_path)
    saved = tmp_path / "saved.json"
    assert main(["--results-dir", str(res), "--test-file", str(test), "--output", str(saved)]) == 0
    assert main(["--results-dir", str(res), "--test-file", str(test), "--check-against", str(saved)]) == 0
    tampered = json.loads(saved.read_text()); tampered["models"]["sft"]["all_rows"]["point"] = 0.5
    saved.write_text(json.dumps(tampered))
    assert main(["--results-dir", str(res), "--test-file", str(test), "--check-against", str(saved)]) == 4
    assert compare(analyze(res, test), saved)


def test_real_test_file_loads_and_has_800_rows():
    rows = load_expected()
    assert len(rows) == 800 and len({g for g, _ in rows.values()}) == 400
