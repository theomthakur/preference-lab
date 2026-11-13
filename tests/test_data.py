import collections, tempfile
from pathlib import Path
from preference_lab import splits
from preference_lab.data import build_groups, build_dataset, render, SEEN_TEMPLATES, HELD_OUT_TEMPLATE
from preference_lab.oracle import Policy, Ticket, decide, verify

def test_generation_is_deterministic():
    a = [r["row_id"] + r["prompt"] for r in build_dataset(7)]
    b = [r["row_id"] + r["prompt"] for r in build_dataset(7)]
    assert a == b
    assert a != [r["row_id"] + r["prompt"] for r in build_dataset(8)]

def test_build_audit_clean_on_disk():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        splits.build(7, out)
        assert splits.audit(out) == []

def test_oracle_label_matches_rendered_required_ids_for_every_row():
    for r in build_dataset(7):
        a, keys, _ = decide(Policy(**r["policy"]), Ticket(**r["ticket"]))
        assert a == r["label"]
        assert sorted(r["id_map"][k] for k in keys) == r["required_ids"]
        assert all(f"[{i}]" in r["prompt"] for i in r["required_ids"])

def test_group_never_crosses_splits_and_sizes():
    g = build_groups(7)
    assert collections.Counter(x["split"] for x in g) == {"sft": 400, "pool": 1000, "val": 200, "test": 400}
    assert len({x["group_id"] for x in g}) == 2000 and len({x["hash"] for x in g}) == 2000

def test_always_hold_cannot_win():
    rows = [r for r in build_dataset(7) if r["split"] == "test" and r["template"] in SEEN_TEMPLATES]
    hold = sum(verify('{"action":"HOLD","evidence":[]}', r["label"], r["required_ids"])["correct"] for r in rows)
    assert hold == 0                                   # empty evidence never scores
    right_action = sum(r["label"] == "HOLD" for r in rows) / len(rows)
    assert right_action < 0.4                          # even a perfect HOLD guess is capped near one third

def test_superseded_and_note_lines_are_never_required():
    for r in build_dataset(7):
        required_text = [l for l in r["prompt"].splitlines() if any(l.startswith(f"[{i}]") for i in r["required_ids"])]
        assert not any("superseded" in l or l.split("] ", 1)[1].startswith("Note:") for l in required_text)

def test_ids_are_not_positionally_fixed():
    keys = collections.Counter()
    for r in build_dataset(7)[:300]:
        keys[r["id_map"]["window"]] += 1
    assert len(keys) >= 8     # the window rule shows up under many different ids

def test_held_out_template_only_in_test_and_labels_match_seen_twin():
    rows = build_dataset(7)
    by = collections.defaultdict(dict)
    for r in rows: by[r["group_id"]][r["template"]] = r
    for gid, d in by.items():
        if HELD_OUT_TEMPLATE in d:
            assert d[HELD_OUT_TEMPLATE]["split"] == "test"
            other = [v for t, v in d.items() if t != HELD_OUT_TEMPLATE][0]
            assert other["label"] == d[HELD_OUT_TEMPLATE]["label"]
