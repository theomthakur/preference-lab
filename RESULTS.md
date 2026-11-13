# Results

One seed, a 0.5B model on a laptop, a synthetic return-policy task. Read every number with that in mind.

## Held-out test (run once, 13 Nov 2026)
The test set has 400 groups (800 rendered rows: 400 with a seen phrasing template and 400 with a held-out template). Each model was frozen first (commit `a1a3e2e`, see `runs/test-results/FREEZE.json`) and run once with the same greedy decoding.

Strict accuracy means the right action AND exactly the right evidence lines. 95% intervals come from a group-level bootstrap (2,000 resamples), because the two rows of a group are not independent.

| Model | Strict correct, all rows | 95% CI | Seen template | Held-out template |
|---|---|---|---|---|
| Base (no training) | 0.000 | 0.000 to 0.000 | 0.000 | 0.000 |
| SFT (5 epochs) | 0.117 | 0.094 to 0.140 | 0.158 | 0.077 |
| DPO, clean pairs | 0.095 | 0.074 to 0.116 | 0.135 | 0.055 |
| DPO, 20% flipped pairs | 0.099 | 0.078 to 0.120 | 0.142 | 0.055 |
| GRPO (32 updates from SFT) | 0.120 | 0.096 to 0.144 | 0.163 | 0.077 |

Paired differences (same test groups, group bootstrap):

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| SFT minus base | +0.118 | +0.094 to +0.140 | Clear gain: the model learned the format and part of the rules |
| DPO clean minus SFT | -0.023 | -0.046 to +0.004 | Leans worse, not conclusive at 95% |
| DPO corrupt minus DPO clean | +0.004 | -0.020 to +0.026 | No detectable effect of flipping 20% of labels |
| GRPO minus SFT | +0.003 | -0.003 to +0.008 | No detectable change |

What the test shows:
- Every trained model is clearly better than the untrained base model.
- Accuracy roughly halves on the held-out phrasing template for every model, so the models partly rely on the wording they were trained on.
- Neither DPO nor GRPO improved on SFT. The GRPO reward signal was almost always flat (88% of groups had no variation), which explains why.
- Corrupted preference labels did not visibly hurt DPO. With this model and data size, the effect is too small to detect.

## Validation (200 cases, used for every decision before the test)
| Model | Strict correct | Right action | Exact evidence |
|---|---|---|---|
| Base | 0 / 200 | 27.5% | 0% |
| SFT, 1 epoch | 0 / 200 | 35% | 1% |
| SFT, 5 epochs | 25 / 200 (12.5%) | 60% | 16.5% |
| DPO, clean | 29 / 200 (14.5%) | 61.5% | 18% |
| DPO, 20% flipped | 26 / 200 (13.0%) | 64.5% | 14% |
| GRPO, 32 updates | 23 / 200 (11.5%) | 59.5% | 15.5% |

The 5-epoch choice was made after seeing the 1-epoch validation result. Both are kept.

## Tool environment (Track B, 20 validation cases)
- Base and SFT both completed the full policy, ticket, finish sequence in most episodes. Neither answered correctly. See `docs/TOOL-ENVIRONMENT.md`.
- Trained tool use was not achieved and is not claimed.

## Other checks
- Hand review of 40 oracle decisions by a human reader: 40 of 40 agree (`docs/HAND-REVIEW.md`).
- Second independently written oracle: agrees on all 1,800 rows (tests).
- 43 automated tests pass.

## Honest notes on the test run
- The first analysis script crashed after all five models had written their predictions. The cause was a bug in my bootstrap code, not in the models. The models were not rerun. The analysis was rewritten to read the saved predictions (`src/preference_lab/test_analysis.py`), and those predictions are the only test outputs.
- Nothing about the models was changed after the test was run.
