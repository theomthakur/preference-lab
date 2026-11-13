from preference_lab.environment import ToolEnv, POLICY_ID, TICKET_ID, MAX_TOOL_CALLS
from preference_lab.agent import run_episode, scripted_cases, from_list, parse
from preference_lab.splits import load

ROW = load("val")[0]
GROUP = {"policy": ROW["policy"], "ticket": ROW["ticket"]}
REQ = [ToolEnv(GROUP).key_ids[k] for k in ROW["oracle_keys"]]

def test_budget_counts_finish_and_wrong_calls():
    env = ToolEnv(GROUP)
    assert env.call("get_policy", {"policy_id": "bad"})["status"] == "wrong_tool_args"
    assert env.call("get_ticket", {"ticket_id": TICKET_ID})["status"] == "ok"
    assert env.call("get_policy", {"policy_id": POLICY_ID})["status"] == "ok"
    assert env.call("finish", {"action": "DENY", "evidence": []})["status"] == "budget_exceeded"

def test_scripted_trajectories_all_match_expected_category():
    for name, faults, outs, expected in scripted_cases(ROW, REQ):
        env = ToolEnv(GROUP, faults)
        r = run_episode(env, from_list(outs), ROW, REQ, {"episode": name})
        assert r["category"] == expected, (name, r["category"])
        if name == "stale_policy_is_reported":
            assert any(c.get("status") == "stale_source" for c in r["calls"])

def test_hidden_answer_not_exposed():
    env = ToolEnv(GROUP)
    assert not hasattr(env, "label") and not hasattr(env, "required_ids")
    text = " ".join(l["text"] for l in env.call("get_policy", {"policy_id": POLICY_ID})["data"]["lines"])
    assert not any(w in text for w in ("ACCEPT", "DENY", "HOLD"))

def test_policy_is_read_only_and_consistent():
    env = ToolEnv(GROUP)
    a = env.call("get_policy", {"policy_id": POLICY_ID})["data"]
    b = env.call("get_policy", {"policy_id": POLICY_ID})["data"]
    assert a == b

def test_episodes_are_independent():
    e1, e2 = ToolEnv(GROUP), ToolEnv(GROUP)
    e1.call("get_ticket", {"ticket_id": TICKET_ID})
    assert e1.calls == 1 and e2.calls == 0

def test_malformed_output_never_crashes():
    for bad in ["", "DENY", "{", '{"tool": 5}', '{"tool": "finish", "action": "DENY"}']:
        r = run_episode(ToolEnv(GROUP), from_list([bad]), ROW, REQ, {})
        assert r["category"] in ("malformed_action", "no_finish") and r["correct"] is False

def test_parse_takes_json_inside_text():
    assert parse('Sure {"tool": "get_ticket", "ticket_id": "ticket-current"} ok')["tool"] == "get_ticket"

def test_parse_uses_first_object_and_ignores_extra_calls():
    two = '{"tool": "get_policy", "policy_id": "return-policy-v1"}, {"tool": "get_ticket", "ticket_id": "ticket-current"}'
    assert parse(two)["tool"] == "get_policy"
