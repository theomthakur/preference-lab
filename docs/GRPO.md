# GRPO (Track A): speed test, bounded run, validation result

## Setup
- Start: the SFT model, with the SFT adapter MERGED into the base weights first (`runs/sft-merged`). TRL's frozen reference is therefore the SFT model, not the untuned base.
- New LoRA on top (r=8, alpha=16, q_proj and v_proj). Learning rate 5e-6, beta 0.04 (KL to the reference), 4 completions per prompt, 2 prompts per update (8 completions per update), up to 96 new tokens, sampled (temperature 1).
- Reward: 1.0 only for a valid answer with the correct action and exactly the required evidence ids; otherwise 0.0. No format or length bonus.
- Data: the training pool only, never validation or test.

## Speed test (10 updates)
- 65 s total, about 6.5 s per update, peak memory 7.2 GB on the laptop. The earlier 45 to 90 minute estimate was mostly setup and debugging, not compute.

## Bounded run (32 updates, the plan's cap)
- 204.1 s, 6.38 s per update, 256 sampled completions in total.
- Mean reward in the first 8 updates: 0.078. In the last 8 updates: 0.078.
- Share of update groups with zero reward variance (nothing to learn from): 0.88 on average.
- Reading: with the SFT model only about 12% correct, most groups of four samples fail together, so GRPO has almost no signal. The run did not learn in 32 updates. This was not fixed by inventing variance, which the plan forbids.

## Validation (200 cases, same decoding as the other models)
| Model | Strict correct | Right action | Exact evidence |
|---|---|---|---|
| SFT (5 epochs) | 25 / 200 (12.5%) | 60.0% | 16.5% |
| GRPO (32 updates from SFT) | 23 / 200 (11.5%) | 59.5% | 15.5% |

Paired on the same cases: GRPO wins 0, SFT wins 2. This is no improvement. It is a valid negative result, and it is reported as one.

## What would be needed for a fair test
- A model that already gets a meaningful share of cases right (a stronger SFT start, or more SFT data), so groups contain both passes and failures.
- More updates, and a check of how the reward variance changes over time.
- A repeat with several seeds before any claim.
