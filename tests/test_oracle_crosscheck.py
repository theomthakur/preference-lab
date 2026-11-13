"""A second, differently structured implementation of the same written rules, compared on every generated row.
It is written from the rule table in the plan (not copied from oracle.py), so a typo in one will show up as a mismatch."""
from preference_lab.data import build_dataset


def alt_decide(p, t):
    if p["final_sale"]:
        return "DENY"
    if p["receipt_required"] and t["receipt"] != "yes":
        return "DENY" if t["receipt"] == "no" else "HOLD"
    d, w, dw = t["days"], p["window"], p["defect_window"]
    faulty_known = t["faulty"] != "unknown"
    if d > dw:
        return "DENY"
    if d > w:                                        # only a faulty item qualifies after the window
        return "ACCEPT" if t["faulty"] == "yes" else ("DENY" if faulty_known else "HOLD")
    if not p["opened_rule"] or t["opened"] == "no":
        return "ACCEPT"
    if t["faulty"] == "yes":                         # opened or unknown-if-opened, but faulty: qualifies either way
        return "ACCEPT"
    if t["opened"] == "unknown":
        return "HOLD"
    return "DENY" if faulty_known else "HOLD"        # opened == yes


def test_alt_oracle_agrees_on_all_rows():
    rows = build_dataset(7)
    bad = [r["row_id"] for r in rows if alt_decide(r["policy"], r["ticket"]) != r["label"]]
    assert bad == [], bad[:5]
