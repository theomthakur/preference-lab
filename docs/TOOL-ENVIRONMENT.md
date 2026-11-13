# Tool environment (Track B): what was built and what it shows

## What it is
- A read-only environment where the policy and the ticket are not in the prompt. The agent must call `get_policy(policy_id)`, `get_ticket(ticket_id)` and `finish(action, evidence)`.
- At most **3 tool calls** per episode, and every call counts, including finish and calls with wrong arguments.
- Each episode gets a fresh environment. The label, the oracle and the true facts are never shown to the agent.
- Faults can be injected: a timeout on a tool, an exception on a tool, or a stale policy version.
- Every episode is written to a trace with its calls, statuses, final answer, category and grade.

## Harness check (scripted, not a model)
Ten scripted trajectories, each checked against its expected category: 2 of 10 correct by design. Categories covered: task success, wrong action, wrong evidence, timeout, tool exception, malformed final answer, unknown tool, budget exceeded (two ways), stale policy. Run `python -m preference_lab.agent --scripted`.

## Real models (20 validation cases, same episodes for both)
| Model | Strict correct | Calls used (mean) | Main outcome |
|---|---|---|---|
| Base (Qwen2.5-0.5B-Instruct) | 0 / 20 | 4.0 | budget_exceeded in 20 |
| SFT (5 epochs, no tool training) | 0 / 20 | 3.0 | wrong_action 13, wrong_evidence 7 |

What the traces show:
- **Base model:** it called `get_policy` correctly, then guessed the ticket id `ticket-return-request-1` instead of the id in the system prompt. Each wrong call used budget, so it never reached an answer. This is a capability failure, and the harness recorded it correctly.
- **SFT model:** it fetched the policy and then finished without fetching the ticket. Its answers were wrong, so it made no use of the ticket facts. Its training never included tools.

These are valid results. Neither model learned tool use, and this run does not establish it. Multi-turn GRPO for tool use was not attempted, because the plan's gate is a 60-minute integration limit and there was no time to test it honestly.

## Not done
- Trained multi-turn tool use (the plan's gated extension).
- Injected timeouts and malformed responses on real model runs (they are covered only by the scripted check).
- A larger real-model run. Twenty cases are a small sample.

Results are in `runs/env-base/`, `runs/env-sft/` and `runs/env-scripted/`.
