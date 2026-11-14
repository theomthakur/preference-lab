import json
from preference_lab.errors import breakdown, line_kinds

PROMPT = "[L1] Policy: window 10 days.\n[L2] Note: customer was polite.\n[L3] (superseded, version 1) Return window: 7 days.\n[L4] Days since purchase: 12.\n[L5] Faulty? yes."
CASE = {"row_id": "c1", "prompt": PROMPT, "required_ids": ["L1", "L4", "L5"], "rule": "late_but_faulty"}

def pred(evidence, action="ACCEPT", label_ok=True):
    out = json.dumps({"action": action, "evidence": evidence}); ev = set(evidence) == {"L1", "L4", "L5"}
    return {"row_id": "c1", "output": out, "action_ok": label_ok, "evidence_ok": ev, "correct": label_ok and ev}

def test_line_kinds():
    assert line_kinds(PROMPT) == {"L1": "fact", "L2": "note", "L3": "superseded", "L4": "fact", "L5": "fact"}

def test_exact_answer_is_strict_correct():
    b = breakdown([pred(["L1", "L4", "L5"])], {"c1": CASE})
    assert b["top"] == {"strict correct": 1} and b["by_required_lines"][3]["strict"] == 1.0

def test_each_kind_of_wrong_evidence_is_named():
    cases = {"c1": CASE}
    assert breakdown([pred(["L1", "L4"])], cases)["evidence_detail"] == {"missed some required lines only": 1}
    d = breakdown([pred(["L1", "L4", "L5", "L2"])], cases)["evidence_detail"]
    assert d["all required plus extra lines"] == 1 and d["cited a note line"] == 1
    d = breakdown([pred(["L1", "L4", "L3"])], cases)["evidence_detail"]
    assert d["cited the superseded line"] == 1 and d["some right, some wrong"] == 1
    d = breakdown([pred(["L2"])], cases)["evidence_detail"]
    assert d["none of the required lines"] == 1

def test_wrong_action_and_unparseable_output_are_separate():
    b = breakdown([pred(["L1", "L4", "L5"], action="DENY", label_ok=False), {"row_id": "c1", "output": "I think DENY", "action_ok": False, "evidence_ok": False, "correct": False}], {"c1": CASE})
    assert b["top"]["wrong action"] == 1 and b["top"]["no valid evidence list"] == 1
