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

## Protocol fix (13 Nov, same session)
The first real runs used a parser that rejected any reply with two calls in it, so 20 of 20 episodes ended as malformed before any tool ran. The base model had copied the format example and put two calls in one reply. The parser now takes the first complete call in a reply and ignores the rest. That rule is fixed and applies to every model, and a test covers it.

With the fixed parser, and the format example added to the prompt (labelled as an example, not about this case):
- Base model (`runs/env-base-v2`): 0 of 20 correct. 16 episodes ran the full policy, ticket, finish sequence, and 4 looped on the ticket tool and ran out of budget.
- SFT model (`runs/env-sft-v2`): 0 of 20 correct. All 20 episodes ran the full policy, ticket, finish sequence, so the protocol works for it too. Its answers were wrong: 13 wrong action, 7 wrong evidence.

The earlier table (0 of 20 for both models) was produced under the broken parser and should be read as a harness failure, not as a model result.
