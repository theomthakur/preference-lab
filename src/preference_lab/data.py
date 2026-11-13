"""Scenario generator, prompt rendering and group-level splits.

One *group* is one underlying case (policy values + ticket facts). A group never crosses a split. Train/val groups
are rendered with the seen phrasing templates 0-2. Test groups are rendered twice: once with a seen template and
once with the held-out template 3, and half of the test groups use policy window values never seen in training.
"""
from __future__ import annotations
import hashlib, json, random
from dataclasses import asdict
from .oracle import Policy, Ticket, decide

GENERATOR_VERSION = "1.0"
TRAIN_WINDOWS = (7, 10, 14, 21, 30)
NOVEL_WINDOWS = (12, 18, 28)
SEEN_TEMPLATES = (0, 1, 2)
HELD_OUT_TEMPLATE = 3
SPLIT_SIZES = {"sft": 400, "pool": 1000, "val": 200, "test": 400}

SYSTEM = ("You decide whether a return request is accepted. Use only the policy and ticket facts below. "
          'Reply with one JSON object: {"action": "ACCEPT" | "DENY" | "HOLD", "evidence": [ids of the lines that decide it]}. '
          "Use HOLD only when a needed fact is unknown. Cite only current policy lines and ticket facts, never notes or superseded lines.")

NOTES = ["Customer mentions the box was a gift.", "Customer asked about shipping times.", "Order was placed on a mobile app.",
         "Customer was polite on the call.", "Customer has bought from us twice before.", "Package arrived on a Tuesday."]


def _yn(v):  # unknown handling for rendering
    return v


# Each template renders a fact key and its value into one line of text.
def _line(template: int, key: str, p: Policy, t: Ticket) -> str:
    cat = "this item"
    opened = {"yes": "opened", "no": "unopened", "unknown": "not stated whether opened"}
    faulty = {"yes": "reported faulty", "no": "reported not faulty", "unknown": "fault status missing"}
    receipt = {"yes": "receipt provided", "no": "no receipt", "unknown": "receipt status missing"}
    fs = "Final sale items cannot be returned" if p.final_sale else "This item is not final sale"
    orule = ("An opened item can be returned only if it is faulty" if p.opened_rule
             else "Opened items can be returned")
    rrule = "A receipt is required" if p.receipt_required else "A receipt is not required"
    T = {
        0: {"final_sale": fs + ".", "window": f"Return window: {p.window} days.", "defect_window": f"Faulty items may be returned up to {p.defect_window} days.",
            "opened_rule": orule + ".", "receipt_rule": rrule + ".", "days": f"Days since purchase: {t.days}.", "opened": f"Condition: {opened[t.opened]}.",
            "faulty": f"Fault: {faulty[t.faulty]}.", "receipt": f"Receipt: {receipt[t.receipt]}."},
        1: {"final_sale": f"Sale type: {'final sale, no returns' if p.final_sale else 'regular'}.", "window": f"Standard returns accepted within {p.window} days of purchase.",
            "defect_window": f"A faulty item is covered for {p.defect_window} days.", "opened_rule": orule + ".", "receipt_rule": rrule + ".",
            "days": f"The customer bought it {t.days} days ago.", "opened": f"The item is {opened[t.opened]}.", "faulty": f"The item is {faulty[t.faulty]}.",
            "receipt": f"{receipt[t.receipt].capitalize()}."},
        2: {"final_sale": fs + " (rule).", "window": f"Within {p.window} days: standard return.", "defect_window": f"Defect returns: {p.defect_window} days maximum.",
            "opened_rule": "Rule on opened goods: " + orule.lower() + ".", "receipt_rule": "Rule on receipts: " + rrule.lower() + ".", "days": f"{t.days} days have passed since purchase.",
            "opened": f"Opened? {t.opened}.", "faulty": f"Faulty? {t.faulty}.", "receipt": f"Receipt? {t.receipt}."},
        3: {"final_sale": f"Policy flag: {'FINAL SALE' if p.final_sale else 'standard sale'}.", "window": f"Policy: returns up to {p.window} d.",
            "defect_window": f"Policy: defective goods up to {p.defect_window} d.", "opened_rule": "Policy: " + ("opened goods need a fault." if p.opened_rule else "opened goods are fine."),
            "receipt_rule": "Policy: " + ("proof of purchase required." if p.receipt_required else "no proof of purchase needed."),
            "days": f"Age of purchase: {t.days} d.", "opened": f"State: {opened[t.opened]}.", "faulty": f"Defect: {faulty[t.faulty]}.", "receipt": f"Proof of purchase: {receipt[t.receipt]}."},
    }
    return T[template][key]


POLICY_KEYS = ["final_sale", "window", "defect_window", "opened_rule", "receipt_rule"]
TICKET_KEYS = ["days", "opened", "faulty", "receipt"]


def render(group: dict, template: int, rng_seed: int) -> dict:
    """Render one group into a prompt. Line order, ids, notes and a superseded line vary with rng_seed."""
    p, t = Policy(**group["policy"]), Ticket(**group["ticket"])
    r = random.Random(rng_seed)
    lines = [(k, _line(template, k, p, t), "current") for k in POLICY_KEYS + TICKET_KEYS]
    for n in r.sample(NOTES, r.choice([0, 1, 2])):
        lines.append((None, f"Note: {n}", "note"))
    if r.random() < 0.35:  # a superseded rule with a different window: must not be cited
        old = p.window + r.choice([-4, 3, 7])
        lines.append((None, f"(superseded, version 1) Return window: {max(old, 1)} days.", "superseded"))
    r.shuffle(lines)
    id_map, shown = {}, []
    for i, (key, text, kind) in enumerate(lines, start=1):
        lid = f"L{i}"
        if key:
            id_map[key] = lid
        shown.append(f"[{lid}] {text}")
    action, keys, rule = decide(p, t)
    required_ids = sorted(id_map[k] for k in keys)
    prompt = "\n".join(shown)
    return {"template": template, "prompt": prompt, "system": SYSTEM, "id_map": id_map, "label": action,
            "required_ids": required_ids, "rule": rule, "oracle_keys": sorted(keys)}


def answer_json(label: str, required_ids: list[str]) -> str:
    return json.dumps({"action": label, "evidence": sorted(required_ids, key=lambda s: int(s[1:]))})


def _sample_group(r: random.Random, windows) -> dict:
    window = r.choice(windows)
    defect = window + r.choice([0, 7, 14, 30])
    p = Policy(final_sale=r.random() < 0.10, window=window, defect_window=defect, opened_rule=r.random() < 0.65, receipt_required=r.random() < 0.45)
    cands = [window - 2, window - 1, window, window + 1, defect - 1, defect, defect + 1, defect + 5, r.randint(1, defect + 20)]
    days = max(1, r.choice(cands))
    st = lambda: r.choices(["yes", "no", "unknown"], weights=[0.4, 0.4, 0.2])[0]
    t = Ticket(days=days, opened=st(), faulty=st(), receipt=r.choices(["yes", "no", "unknown"], weights=[0.55, 0.25, 0.2])[0])
    return {"policy": asdict(p), "ticket": asdict(t)}


def group_hash(g: dict) -> str:
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


def build_groups(seed: int = 7) -> list[dict]:
    """Generate all groups, balanced to roughly one third each label, globally unique, then assign splits."""
    r = random.Random(seed)
    seen, out = set(), []
    for split, n in SPLIT_SIZES.items():
        counts = {a: 0 for a in ("ACCEPT", "DENY", "HOLD")}
        quota = {"ACCEPT": n // 3 + n % 3, "DENY": n // 3, "HOLD": n // 3}
        novel = 0
        while sum(counts.values()) < n:
            use_novel = split == "test" and novel < n // 2
            g = _sample_group(r, NOVEL_WINDOWS if use_novel else TRAIN_WINDOWS)
            h = group_hash(g)
            action = decide(Policy(**g["policy"]), Ticket(**g["ticket"]))[0]
            if h in seen or counts[action] >= quota[action]:
                continue
            seen.add(h); counts[action] += 1; novel += use_novel
            out.append({"group_id": f"g{len(out):05d}", "split": split, "hash": h, "novel_values": use_novel, **g})
    return out


def build_dataset(seed: int = 7) -> list[dict]:
    """One row per rendered prompt. Train/val: one seen-template rendering per group. Test: seen + held-out template."""
    rows = []
    for g in build_groups(seed):
        templates = [SEEN_TEMPLATES[int(g["hash"], 16) % 3]]
        if g["split"] == "test":
            templates.append(HELD_OUT_TEMPLATE)
        for tpl in templates:
            rr = render(g, tpl, rng_seed=int(g["hash"], 16) + tpl)
            rows.append({"row_id": f"{g['group_id']}-t{tpl}", "group_id": g["group_id"], "split": g["split"], "group_hash": g["hash"],
                         "novel_values": g["novel_values"], "policy": g["policy"], "ticket": g["ticket"], "generator_version": GENERATOR_VERSION, **rr})
    return rows
