import json, statistics
from preference_lab.pairs import build_pairs
from preference_lab.oracle import verify
from preference_lab.splits import load

def test_pairs_valid_and_pool_only():
    clean, corrupt, flags = build_pairs()
    pool_ids = {r["row_id"] for r in load("pool")}
    held = {r["row_id"] for s in ("val", "test", "sft") for r in load(s)}
    rows = {r["row_id"]: r for r in load("pool")}
    assert len(clean) == 1000 and {p["row_id"] for p in clean} == pool_ids and not ({p["row_id"] for p in clean} & held)
    for p in clean:
        r = rows[p["row_id"]]
        assert p["chosen"] != p["rejected"]
        assert verify(p["chosen"], r["label"], r["required_ids"])["correct"]
        assert not verify(p["rejected"], r["label"], r["required_ids"])["correct"]
        json.loads(p["rejected"])                       # rejected is valid JSON, so format cannot separate the pair

def test_corruption_is_a_seeded_swap_of_20_percent_and_flags_are_hidden():
    clean, corrupt, flags = build_pairs()
    flipped = {c["row_id"] for c, k in zip(clean, corrupt) if c["chosen"] != k["chosen"]}
    assert flipped == set(flags["corrupted_row_ids"]) and len(flipped) == 200
    assert all("corrupt" not in "".join(k.keys()) for k in corrupt)       # no flag field inside learner data
    assert build_pairs()[2] == flags                                    # deterministic

def test_no_length_cue():
    clean, _, _ = build_pairs()
    d = [len(p["rejected"]) - len(p["chosen"]) for p in clean]
    assert abs(statistics.mean(d)) < 8                                  # chars; the cue stays small
