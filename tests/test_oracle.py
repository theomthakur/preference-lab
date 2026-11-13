import pytest
from preference_lab.oracle import Policy, Ticket, decide, verify, parse_answer

P = dict(final_sale=False, window=14, defect_window=30, opened_rule=True, receipt_required=False)
T = dict(days=10, opened="yes", faulty="no", receipt="yes")

def run(p=None, t=None):
    return decide(Policy(**{**P, **(p or {})}), Ticket(**{**T, **(t or {})}))

CASES = [
    ({"final_sale": True}, {}, "DENY", "final_sale"),
    ({"receipt_required": True}, {"receipt": "no"}, "DENY", "no_receipt"),
    ({"receipt_required": True}, {"receipt": "unknown"}, "HOLD", "receipt_unknown"),
    ({"receipt_required": False}, {"receipt": "no"}, "DENY", "opened_not_faulty"),   # receipt ignored when not required
    ({}, {"days": 31}, "DENY", "too_late"),
    ({}, {"days": 20, "faulty": "yes"}, "ACCEPT", "late_but_faulty"),
    ({}, {"days": 20, "faulty": "no"}, "DENY", "late_not_faulty"),
    ({}, {"days": 20, "faulty": "unknown"}, "HOLD", "late_fault_unknown"),
    ({}, {"days": 14, "opened": "no"}, "ACCEPT", "in_window_unopened"),           # boundary: day 14 is inside
    ({}, {"days": 15, "opened": "no", "faulty": "no"}, "DENY", "late_not_faulty"),                     # boundary: day 15 is outside
    ({}, {"days": 10, "opened": "yes", "faulty": "no"}, "DENY", "opened_not_faulty"),   # the plan's example
    ({}, {"days": 10, "opened": "yes", "faulty": "unknown"}, "HOLD", "opened_fault_unknown"),
    ({}, {"days": 10, "opened": "yes", "faulty": "yes"}, "ACCEPT", "opened_faulty"),
    ({"opened_rule": False}, {"opened": "yes", "faulty": "no"}, "ACCEPT", "in_window"),
    ({}, {"opened": "unknown", "faulty": "yes"}, "ACCEPT", "faulty_regardless_of_opened"),
    ({}, {"opened": "unknown", "faulty": "no"}, "HOLD", "opened_unknown"),
    ({"final_sale": True, "receipt_required": True}, {"receipt": "unknown"}, "DENY", "final_sale"),  # priority: final sale first
]

@pytest.mark.parametrize("p,t,action,rule", CASES)
def test_oracle_cases(p, t, action, rule):
    a, ev, r = run(p, t)
    assert (a, r) == (action, rule)
    assert len(ev) >= 1

def test_changing_the_policy_changes_the_answer():
    assert run({"window": 14}, {"days": 10, "opened": "no"})[0] == "ACCEPT"
    assert run({"window": 7, "defect_window": 7}, {"days": 10, "opened": "no"})[0] == "DENY"
    assert run({"opened_rule": True})[0] == "DENY" and run({"opened_rule": False})[0] == "ACCEPT"

def test_defect_window_never_below_window_is_respected_at_boundary():
    assert run({"window": 14, "defect_window": 14}, {"days": 15, "faulty": "yes"})[0] == "DENY"

# ---- verifier / reward-gaming ----
def ok(action, ids): return f'{{"action": "{action}", "evidence": {ids}}}'

def test_verifier_requires_action_and_exact_evidence():
    assert verify(ok("DENY", '["L1","L2"]'), "DENY", ["L1", "L2"])["correct"]
    assert not verify(ok("ACCEPT", '["L1","L2"]'), "DENY", ["L1", "L2"])["correct"]       # valid JSON, wrong decision
    assert not verify(ok("DENY", '["L1"]'), "DENY", ["L1", "L2"])["correct"]              # missing evidence
    assert not verify(ok("DENY", '["L1","L2","L9"]'), "DENY", ["L1", "L2"])["correct"]    # extra / fabricated evidence
    assert not verify(ok("DENY", '["L1","L1","L2"]'), "DENY", ["L1", "L2"])["correct"]    # duplicates

def test_verifier_rejects_garbage_and_never_crashes():
    for bad in ["", None, "DENY", "{", "[]", '{"action": 3, "evidence": []}', '{"action": "DENY"}', '{"action":"DENY","evidence":"L1"}']:
        r = verify(bad, "DENY", ["L1"])
        assert r["correct"] is False
    assert not verify(ok("MAYBE", '["L1"]'), "DENY", ["L1"])["valid"]

def test_verifier_tolerates_json_inside_text_but_scores_by_content():
    r = verify('Sure! {"action": "DENY", "evidence": ["L1"]} thanks', "DENY", ["L1"])
    assert r["correct"]
    assert parse_answer("no json here") is None

def test_correct_action_without_evidence_is_visible_but_not_correct():
    r = verify('{"action": "DENY"}', "DENY", ["L1"])
    assert r["action_ok"] and r["valid"] and not r["format_ok"] and not r["evidence_ok"] and not r["correct"]
