# Final status (13 Nov 2026, after the review pass)

## Completed and evaluated
- Task spec, rule oracle, strict verifier
- Generator, grouped splits, leakage audit (clean)
- Second independent oracle (agrees on all rows), 40-row hand review (40 of 40)
- Shared evaluator; base, SFT (1 and 5 epochs), DPO clean, DPO corrupted, GRPO evaluated on validation
- Read-only tool environment with traces; 10 scripted trajectories; real-model episodes on 20 validation cases
- Frozen held-out test, run once for all five models. The analysis step of that script crashed after the predictions were saved; the metrics were recomputed from the saved predictions with no model rerun.
- RESULTS.md (with statistical notes), DATA-CARD.md, MODEL-CARD.md, WALKTHROUGH.md, demo, architecture diagram, artifact inventory (`docs/ARTIFACTS.md`)

## Review pass (what changed and why)
- README: removed the claim that the held-out test had not been touched; added the held-out table and explained the crashed analysis step.
- Install: `uv pip install -e .` did not use the lock file. The documented commands are now `uv pip sync requirements.lock.txt` then `uv pip install -e . --no-deps`, or the plain-pip equivalent.
- Documented which artifacts are in GitHub and which are not (weights), and exactly what a fresh clone can run.
- Statistical reporting: explained why a bootstrap interval for zero successes is [0, 0], added a supplementary exact upper bound, and stated that the GRPO flat-reward explanation is a hypothesis, not a finding.
- `test_analysis.py` now validates its input against `data/test.jsonl` (missing, duplicated, unexpected and misaligned rows are errors), prints by default, refuses to overwrite files, and can check a recomputation against the saved metrics. Tests added.
- Found while reviewing tracked files: `grpo.py` skipped building the merged SFT model whenever `config.json` existed, and `runs/sft-merged/config.json` is tracked but the weights are not. A fresh clone would have failed or loaded nothing useful. It now checks for the weights and fails fast with a clear message. Tested.
- No model was retrained or rerun, and no result file was changed.

## Verified in a clean checkout of commit f2c7392 (temporary clone, not my working folder)
- Install from the lock file by uv and by plain pip: the installed set equals the 53 pinned packages (pip: after normalizing package-name spelling).
- 59 tests pass. Leakage audit clean.
- `splits build` and `pairs` regenerate `data/` byte for byte (no tracked data file changed).
- `demo` runs from tracked predictions. `test_analysis --check-against` reproduces the saved metrics exactly. The scripted tool harness reproduces the tracked traces.
- `grpo` fails fast with a clear message when weights are missing. `final_test` refuses to run.

## Incomplete or not verified
- Trained multi-turn tool use (the plan's gated step was not reached)
- Real-model tool episodes beyond 20 validation cases
- Multi-seed runs (one training seed per method)
- Retraining from scratch on another machine has not been verified to give the same numbers. The trained weights are not in GitHub.
- Linux, CUDA and CPU were not tested. The lock file was produced on macOS arm64.
- Known housekeeping: `src/preference_lab.egg-info/` is tracked in git and changes when the package is installed. Untracking it is proposed, not done.
- A read-only dashboard (`web/`, `vercel.json`) was committed by a separate tool session while this review was running (commit `b4e4207`). It was not created or reviewed in this pass. Its `data.json` is derived from saved results and has not been checked against `test_metrics.json`. This pass made no deployment.

## Optional, not done
- Prompt-injection robustness slice
- Oracle-filter vs random-removal arm on corrupted data

## Not claimed
- Calibration (no defensible probabilities)
- Any improvement from DPO or GRPO over SFT
- Frontier-scale training, research track record, or production use

## Future work (would need a new validation set and a separately frozen test)
- A stronger SFT start, so GRPO groups contain both passes and failures, then a controlled comparison of GRPO settings
- Several training seeds per method, with intervals that include training variance
- Tool-use training, with the plan's 60-minute integration gate
