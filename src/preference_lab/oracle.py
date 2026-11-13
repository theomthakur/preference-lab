"""The rule oracle and the answer verifier for the fictional return-policy task.

The oracle decides ACCEPT / DENY / HOLD from structured facts only. It never sees rendered text, so it is
independent of the prompt templates. Evidence is a set of fact KEYS (not prompt ids); data.py maps keys to the
ids shown in a given prompt.

Decision order (first match wins):
  1. policy.final_sale                          -> DENY   evidence {final_sale}
  2. policy.receipt_required
       receipt == no                            -> DENY   {receipt_rule, receipt}
       receipt == unknown                       -> HOLD   {receipt_rule, receipt}
  3. days > defect_window                       -> DENY   {window, defect_window, days}
  4. window < days <= defect_window  (only faulty items qualify)
       faulty == yes                            -> ACCEPT {defect_window, days, faulty}
       faulty == no                             -> DENY   {window, days, faulty}
       faulty == unknown                        -> HOLD   {defect_window, days, faulty}
  5. days <= window
       opened_rule is False (opened is fine)    -> ACCEPT {window, days}
       opened == no                             -> ACCEPT {window, days, opened}
       opened == yes: faulty yes -> ACCEPT {window, opened_rule, opened, faulty}
                      faulty no  -> DENY   {window, opened_rule, opened, faulty}
                      faulty unknown -> HOLD {opened_rule, opened, faulty}
       opened == unknown: faulty yes -> ACCEPT {window, opened_rule, faulty}   (qualifies either way)
                          otherwise    -> HOLD   {opened_rule, opened}
"""
from __future__ import annotations
import json
from dataclasses import dataclass

ACTIONS = ("ACCEPT", "DENY", "HOLD")


@dataclass(frozen=True)
class Policy:
    final_sale: bool
    window: int            # days a normal return is allowed
    defect_window: int     # days a faulty item can still be returned (>= window)
    opened_rule: bool      # True: an opened item is returnable only if faulty
    receipt_required: bool


@dataclass(frozen=True)
class Ticket:
    days: int
    opened: str            # "yes" | "no" | "unknown"
    faulty: str
    receipt: str


def decide(p: Policy, t: Ticket) -> tuple[str, frozenset[str], str]:
    """Return (action, evidence_keys, rule_name)."""
    if p.final_sale:
        return "DENY", frozenset({"final_sale"}), "final_sale"
    if p.receipt_required:
        if t.receipt == "no":
            return "DENY", frozenset({"receipt_rule", "receipt"}), "no_receipt"
        if t.receipt == "unknown":
            return "HOLD", frozenset({"receipt_rule", "receipt"}), "receipt_unknown"
    if t.days > p.defect_window:
        return "DENY", frozenset({"window", "defect_window", "days"}), "too_late"
    if t.days > p.window:
        if t.faulty == "yes":
            return "ACCEPT", frozenset({"defect_window", "days", "faulty"}), "late_but_faulty"
        if t.faulty == "no":
            return "DENY", frozenset({"window", "days", "faulty"}), "late_not_faulty"
        return "HOLD", frozenset({"defect_window", "days", "faulty"}), "late_fault_unknown"
    if not p.opened_rule:
        return "ACCEPT", frozenset({"window", "days"}), "in_window"
    if t.opened == "no":
        return "ACCEPT", frozenset({"window", "days", "opened"}), "in_window_unopened"
    if t.opened == "yes":
        keys = frozenset({"window", "opened_rule", "opened", "faulty"})
        if t.faulty == "yes":
            return "ACCEPT", keys, "opened_faulty"
        if t.faulty == "no":
            return "DENY", keys, "opened_not_faulty"
        return "HOLD", frozenset({"opened_rule", "opened", "faulty"}), "opened_fault_unknown"
    # opened unknown
    if t.faulty == "yes":
        return "ACCEPT", frozenset({"window", "opened_rule", "faulty"}), "faulty_regardless_of_opened"
    return "HOLD", frozenset({"opened_rule", "opened"}), "opened_unknown"


def parse_answer(text: str) -> dict | None:
    """Parse the model output. Accepts a bare JSON object, or one JSON object inside surrounding text.
    Returns None when no valid {"action": ..., "evidence": [...]} object is present."""
    text = (text or "").strip()
    cands = [text]
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        cands.append(text[a:b + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("action"), str) and isinstance(obj.get("evidence"), list) \
                and all(isinstance(x, str) for x in obj["evidence"]):
            return obj
    return None


def verify(text: str, label: str, required_ids: list[str]) -> dict:
    """Primary correctness: 1 only for valid output, correct action AND exactly the required evidence ids.
    Format validity, action correctness and evidence correctness are also reported separately."""
    obj = parse_answer(text)
    if obj is None:
        return {"valid": False, "action_ok": False, "evidence_ok": False, "correct": False, "action": None}
    action = obj["action"].strip().upper()
    action_ok = action == label
    evidence_ok = set(obj["evidence"]) == set(required_ids) and len(obj["evidence"]) == len(set(obj["evidence"]))
    return {"valid": action in ACTIONS, "action_ok": action_ok, "evidence_ok": evidence_ok,
            "correct": action_ok and evidence_ok, "action": action}
