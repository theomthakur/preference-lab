# Final status (13 Nov 2026)

## Completed and evaluated
- Task spec, rule oracle, strict verifier, 43 tests
- Generator, grouped splits, leakage audit (clean)
- Second independent oracle (agrees on all rows), 40-row hand review (40 of 40)
- Shared evaluator, base, SFT (1 and 5 epochs), DPO clean, DPO corrupted, GRPO: validation results
- Read-only tool environment with traces; 10 scripted trajectories; real-model episodes on 20 validation cases
- Frozen held-out test, run once for all five models, with intervals and paired comparisons
- RESULTS.md, DATA-CARD.md, MODEL-CARD.md, WALKTHROUGH.md, demo script, architecture diagram in the README

## Incomplete
- Trained multi-turn tool use (the plan's gated step was not reached)
- Real-model tool episodes beyond 20 validation cases
- Multi-seed runs (one seed only)
- Clean-checkout verification (see below)

## Optional, not done
- Prompt-injection robustness slice
- Oracle-filter vs random-removal arm on corrupted data
- Read-only viewer for saved predictions

## Not claimed
- Calibration (no defensible probabilities)
- Any improvement from DPO or GRPO over SFT
- Frontier-scale training, research track record, or production use
