"""Track B: a read-only tool environment. The policy and the ticket are NOT in the prompt; the agent must fetch them.

Tools:  get_policy(policy_id), get_ticket(ticket_id), finish(action, evidence)
Limit:  at most MAX_TOOL_CALLS calls per episode, counting finish and any call with wrong arguments.
Hidden: the label, the oracle and the true facts are never exposed to the agent. Each episode gets a fresh ToolEnv.
Faults can be injected per episode: timeout_on=<tool>, exception_on=<tool>, stale=True (the policy returns an old window).
"""
from __future__ import annotations
from .data import POLICY_KEYS, TICKET_KEYS, _line
from .oracle import Policy, Ticket

MAX_TOOL_CALLS = 3
POLICY_ID = "return-policy-v1"
TICKET_ID = "ticket-current"


class ToolEnv:
    def __init__(self, group: dict, faults: dict | None = None):
        self._p = Policy(**group["policy"])
        self._t = Ticket(**group["ticket"])
        self._faults = dict(faults or {})
        self.key_ids: dict[str, str] = {}       # fact key -> line id shown to the agent (used for grading only)
        self._policy: list[tuple[str, str, str]] = []  # (key, id, text)
        self._ticket: list[tuple[str, str, str]] = []
        n = 0
        for k in POLICY_KEYS:
            n += 1; lid = f"L{n}"; self.key_ids[k] = lid
            self._policy.append((k, lid, _line(1, k, self._p, self._t)))
        for k in TICKET_KEYS:
            n += 1; lid = f"L{n}"; self.key_ids[k] = lid
            self._ticket.append((k, lid, _line(1, k, self._p, self._t)))
        self.calls = 0

    def _budget_ok(self) -> bool:
        self.calls += 1
        return self.calls <= MAX_TOOL_CALLS

    def call(self, tool: str, args: dict) -> dict:
        if not self._budget_ok():
            return {"status": "budget_exceeded", "error": f"more than {MAX_TOOL_CALLS} tool calls"}
        if self._faults.get("timeout_on") == tool:
            return {"status": "timeout", "error": "tool timed out"}
        if self._faults.get("exception_on") == tool:
            return {"status": "tool_exception", "error": "simulated backend exception"}
        if tool == "get_policy":
            if args.get("policy_id") != POLICY_ID:
                return {"status": "wrong_tool_args", "error": f"unknown policy_id; use {POLICY_ID}"}
            stale = bool(self._faults.get("stale"))
            lines = []
            for k, lid, text in self._policy:
                if stale and k == "window":
                    text = f"(superseded, version 1) Return window: {self._p.window + 7} days."
                lines.append({"id": lid, "text": text})
            return {"status": "stale_source" if stale else "ok",
                    "data": {"policy_version": 0 if stale else 2, "lines": lines}}
        if tool == "get_ticket":
            if args.get("ticket_id") != TICKET_ID:
                return {"status": "wrong_tool_args", "error": f"unknown ticket_id; use {TICKET_ID}"}
            return {"status": "ok", "data": {"lines": [{"id": lid, "text": text} for _, lid, text in self._ticket]}}
        if tool == "finish":
            return {"status": "ok"}
        return {"status": "wrong_tool_args", "error": f"unknown tool {tool}"}
