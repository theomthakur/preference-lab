# Results

**Later 1.5B follow-up:** A separate SFT continuation reached 48.25% strict accuracy and 84.25% action accuracy on a new 800-row synthetic frozen test. This is not a controlled comparison with the original 0.5B results below. See [the follow-up record](docs/FOLLOWUP-FROZEN-TEST.md).

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
- Neither DPO nor GRPO improved on SFT. DPO clean scored slightly lower than SFT (-0.023, CI -0.046 to +0.004), which is not conclusive either way.
- GRPO's reward was almost always flat and 88% of its update groups had no variation in reward, so most updates carried little learning signal. That is a plausible contributor to its lack of improvement, but it was not tested as the cause. The small learning rate (5e-6), the small number of updates (32), the small LoRA and the KL term could each matter, and no experiment here separates them.
- No difference was detected between DPO on clean and on 20%-flipped labels. This test cannot rule out a small effect in either direction.

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
- 88 automated tests pass, including tests of the saved-prediction analysis (missing, duplicated and misaligned rows, grouped resampling, and the no-overwrite rule) and of the dashboard data and static files.

## Honest notes on the test run
- The first analysis script crashed after all five models had written their predictions. The cause was a bug in my bootstrap code, not in the models. The models were not rerun. The analysis was rewritten to read the saved predictions (`src/preference_lab/test_analysis.py`), and those predictions are the only test outputs.
- Nothing about the models was changed after the test was run.
- The analysis script was later rewritten to validate its inputs against `data/test.jsonl` and to refuse to overwrite files by default. Recomputing from the saved predictions with the rewritten script reproduces the saved `test_metrics.json` exactly (`--check-against`), and the SHA-256 of the prediction and metrics files was unchanged by the review.

## Statistical notes
1. **What the intervals cover.** They are a percentile bootstrap over the 400 test groups (2,000 resamples, seed 7). They reflect which test cases happened to be drawn. They do not include training variance: each method was trained once with one seed, so a difference between methods could partly be training noise that no interval shows.
2. **Zero successes.** For the base model every resample has zero successes, so the percentile interval is [0, 0] and carries no information about uncertainty. A supplementary exact (Clopper-Pearson) two-sided 95% upper bound for 0 of 800 rows is 0.46% if the rows were independent and 0.92% if the 400 groups are treated as the units. The second is the more cautious reading, since two rows of a group are not independent. This is added arithmetic for interpretation; it does not replace the saved metrics. `python -m preference_lab.test_analysis` prints it.
3. **Paired comparisons.** All models were scored on the same test items, so differences are paired. Four comparisons are shown with no multiple-comparison adjustment. The SFT-versus-base gap is not in doubt. A borderline interval such as DPO clean minus SFT (-0.046 to +0.004) should not be read as "worse" or as "equal".
4. **Held-out template.** Each group is rendered with a seen template and a held-out template, so the drop on the held-out template is a within-group comparison. It is a descriptive observation of the same models, consistent in direction across all of them, not a tested effect.
5. **Selection on validation.** The 5-epoch SFT choice was made after seeing the 1-epoch validation result, so validation numbers are not clean estimates for that choice. The test numbers are, because the test was not used for any choice.
6. **Sample size.** Differences of one or two percentage points on 800 rows are well inside the noise of this setup.
