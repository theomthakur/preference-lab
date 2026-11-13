"""Episodes against the read-only environment: a scripted harness check (not a model) and real-model attempts.

  python -m preference_lab.agent --scripted                                   # 10 scripted trajectories, checks the harness
  python -m preference_lab.agent --split val --limit 20 --name env-base       # real base model
  python -m preference_lab.agent --split val --limit 20 --name env-sft --adapter runs/sft/adapter

Every episode is written to runs/<name>/traces.jsonl with the calls, statuses, final answer, category and grade.
Categories: task_success, wrong_action, wrong_evidence, malformed_action, timeout, tool_exception, budget_exceeded, stale_source, no_finish.
"""
from __future__ import annotations
import argparse, collections, json, time
from pathlib import Path
from .environment import ToolEnv, POLICY_ID, TICKET_ID, MAX_TOOL_CALLS
from .oracle import verify
from .splits import load

MAX_TURNS = 6
TOOLS = {"get_policy", "get_ticket", "finish"}
SYSTEM = (
    "You decide a customer return request. The policy and ticket are not shown to you: fetch them with tools. "
    "You have at most 3 tool calls in total, and finish counts as one. Reply with exactly one JSON object per turn, nothing else. "
    f'Tools: {{"tool": "get_policy", "policy_id": "{POLICY_ID}"}}, {{"tool": "get_ticket", "ticket_id": "{TICKET_ID}"}}, '
    '{"tool": "finish", "action": "ACCEPT" | "DENY" | "HOLD", "evidence": ["L..", ...]}. '
    "Evidence lists the line ids that decide the answer. Use HOLD only when a needed fact is unknown."
)
TASK = "Decide this return request. Fetch the policy and the ticket, then finish."


def parse(raw):
    if not raw:
        return None
    s = raw.strip(); a, b = s.find("{"), s.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(s[a:b + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def run_episode(env, next_output, row, required, meta):
    history, calls, final, category = [], [], None, None
    for _ in range(MAX_TURNS):
        raw = next_output(history)
        if raw is None:
            break
        history.append(("assistant", raw))
        obj = parse(raw)
        if obj is None or obj.get("tool") not in TOOLS:
            calls.append({"tool": None, "status": "malformed_action", "raw": raw[:160]}); category = "malformed_action"; break
        tool = obj["tool"]; args = {k: v for k, v in obj.items() if k != "tool"}
        if tool == "finish":
            if not (isinstance(obj.get("action"), str) and isinstance(obj.get("evidence"), list)
                    and all(isinstance(x, str) for x in obj["evidence"])):
                calls.append({"tool": "finish", "args": args, "status": "malformed_action"}); category = "malformed_action"; break
            res = env.call("finish", {"action": obj["action"], "evidence": obj["evidence"]})
            calls.append({"tool": "finish", "args": args, "status": res["status"]})
            if res["status"] == "ok":
                final = (obj["action"], obj["evidence"])
            else:
                category = res["status"]
            break
        res = env.call(tool, args)
        calls.append({"tool": tool, "args": args, "status": res["status"]})
        history.append(("tool", tool, res))
        if res["status"] in ("budget_exceeded", "timeout", "tool_exception"):
            category = res["status"]; break
    correct = False
    if final is not None:
        v = verify(json.dumps({"action": final[0], "evidence": final[1]}), row["label"], required)
        correct = v["correct"]
        category = "task_success" if correct else ("wrong_action" if not v["action_ok"] else "wrong_evidence")
    if category is None:
        category = "no_finish"
    if not correct and any(c.get("status") == "stale_source" for c in calls) and final is not None:
        category = "stale_source"
    return {**meta, "calls": calls, "calls_used": env.calls, "final": list(final) if final else None,
            "category": category, "correct": correct, "label": row["label"], "required_env_ids": required}


def from_list(outputs):
    it = iter(outputs)
    return lambda history: next(it, None)


def scripted_cases(row, required):
    """Ten scripted trajectories. They test the harness; they are not a model. Returns (name, faults, outputs, expected category)."""
    label = row["label"]; wrong = next(a for a in ("ACCEPT", "DENY", "HOLD") if a != label)
    P = json.dumps({"tool": "get_policy", "policy_id": POLICY_ID})
    T = json.dumps({"tool": "get_ticket", "ticket_id": TICKET_ID})
    fin = lambda a, ev: json.dumps({"tool": "finish", "action": a, "evidence": ev})
    return [
        ("correct", {}, [P, T, fin(label, required)], "task_success"),
        ("wrong_action", {}, [P, T, fin(wrong, required)], "wrong_action"),
        ("missing_evidence", {}, [P, T, fin(label, required[:-1])], "wrong_evidence"),
        ("timeout_on_ticket", {"timeout_on": "get_ticket"}, [P, T], "timeout"),
        ("exception_on_policy", {"exception_on": "get_policy"}, [P, T], "tool_exception"),
        ("malformed_final_text", {}, [P, T, f"I think it is {label} because of the window."], "malformed_action"),
        ("unknown_tool", {}, [json.dumps({"tool": "search", "query": "policy"})], "malformed_action"),
        ("four_calls_over_budget", {}, [P, T, P, T], "budget_exceeded"),
        ("wrong_policy_id_costs_budget", {}, [json.dumps({"tool": "get_policy", "policy_id": "other"}), P, T, fin(label, required)], "budget_exceeded"),
        ("stale_policy_is_reported", {"stale": True}, [P, T, fin(label, required)], "task_success"),  # stale call recorded in calls; this answer is still right
    ]


def summarize(recs):
    cats = collections.Counter(r["category"] for r in recs)
    n = len(recs)
    return {"episodes": n, "strict_correct": sum(r["correct"] for r in recs), "categories": dict(cats),
            "mean_calls": round(sum(r["calls_used"] for r in recs) / max(n, 1), 2),
            "over_budget_rate": round(sum(r["calls_used"] > MAX_TOOL_CALLS for r in recs) / max(n, 1), 3)}


def write(out: Path, recs):
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n")
    s = summarize(recs); (out / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripted", action="store_true"); ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=20); ap.add_argument("--name", default="env-scripted")
    ap.add_argument("--adapter")
    a = ap.parse_args()
    rows = load(a.split)
    if a.scripted:
        row = rows[0]; group = {"policy": row["policy"], "ticket": row["ticket"]}
        probe = ToolEnv(group); required = [probe.key_ids[k] for k in row["oracle_keys"]]
        recs = []
        for name, faults, outs, expected in scripted_cases(row, required):
            env = ToolEnv(group, faults)
            r = run_episode(env, from_list(outs), row, required, {"episode": name, "row_id": row["row_id"], "model": "scripted-not-a-model"})
            r["expected"] = expected; r["ok"] = r["category"] == expected and (name != "stale_policy_is_reported" or any(c.get("status") == "stale_source" for c in r["calls"]))
            recs.append(r)
            print(f"{name:32s} expected {expected:16s} got {r['category']:16s} {'OK' if r['ok'] else 'MISMATCH'}")
        print(json.dumps(write(Path("runs") / a.name, recs), indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from .evaluate import MODEL, REVISION, device
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(dev)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    def model_fn(history):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
        for h in history:
            if h[0] == "assistant":
                msgs.append({"role": "assistant", "content": h[1]})
            else:
                msgs.append({"role": "user", "content": "TOOL RESULT: " + json.dumps({"tool": h[1], "result": h[2]})})
        enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(dev)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    recs = []; t0 = time.time()
    for row in rows[: a.limit]:
        group = {"policy": row["policy"], "ticket": row["ticket"]}
        env = ToolEnv(group)
        required = [env.key_ids[k] for k in row["oracle_keys"]]
        recs.append(run_episode(env, model_fn, row, required, {"episode": row["row_id"], "row_id": row["row_id"], "model": a.adapter or "base", "split": a.split}))
    s = write(Path("runs") / a.name, recs); s["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(s, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
