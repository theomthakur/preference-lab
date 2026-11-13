"""Five-minute local demo. Reads saved artifacts only; no model is loaded, so it is safe on the laptop.

  python -m preference_lab.demo

Shows: one normal correct answer, one unsupported answer (evidence the prompt does not contain), and one failure trace from the tool environment.
"""
import json
from pathlib import Path

def first(path, pred):
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if pred(r):
            return r
    return None

def show(title, r, extra=""):
    print(f"\n== {title} ==")
    print(f"row {r['row_id']} (template {r['template']}, rule '{r['rule']}')")
    print(f"correct answer: {r['label']} with evidence {r['required_ids']}")
    print(f"model output  : {r['output']}")
    if extra: print(extra)

test_path = "runs/test-results/predictions_sft.jsonl"
normal = first(test_path, lambda r: r["correct"] and r["template"] == 0)
show("1. Normal case: right action, exact evidence", normal, "verdict: correct (strict)")

unsupported = first(test_path, lambda r: (not r["correct"]) and r["action_ok"] and r["format_ok"] and not r["evidence_ok"])
if unsupported:
    import re
    ids = set(re.findall(r"^\[(L\d+)\]", unsupported.get("prompt", ""), re.M))
    show("2. Wrong evidence: right action, evidence lines do not match the rule", unsupported,
         "verdict: action right, evidence wrong. Strict scoring counts this as wrong.")
trace = first("runs/env-sft-v2/traces.jsonl", lambda r: r["category"] == "wrong_action")
if trace:
    print("\n== 3. Failure trace from the tool environment ==")
    print(f"episode {trace['episode']}, category {trace['category']}, calls used {trace['calls_used']} of 3")
    for c in trace["calls"]:
        print(f"  tool {c['tool']:11s} status {c['status']:10s} args {json.dumps(c.get('args', {}))}")
    print(f"  final {trace['final']}; correct answer {trace['label']} with ids {trace['required_env_ids']}")
print("\nHeld-out test (400 groups, one seed), strict correct:")
m = json.load(open("runs/test-results/test_metrics.json"))["models"]
for k in ("base", "sft", "dpo_clean", "dpo_corrupt", "grpo"):
    v = m[k]["all_rows"]; print(f"  {k:12s} {v['point']:.3f}  95% CI {v['ci95']}")
