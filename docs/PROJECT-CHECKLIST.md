# Project checklist (updated 13 Nov 2026)

Tick items as they are finished. "Heat" is how hard the item works the laptop.

## Done
- [x] Environment, pinned versions, benchmark
- [x] Task spec, rule oracle, strict verifier, 35 tests
- [x] Generator, grouped splits, leakage audit
- [x] Base baseline, SFT (1 and 5 epochs), clean DPO, corrupted DPO, all evaluated on validation
- [x] Preference pairs, clean and 20% corrupted
- [x] README with results and run commands
- [x] 1. Evaluate corrupted DPO
- [x] 2. Compare clean vs corrupted
- [x] 3. Hand-review 40 examples (40 of 40 agree)

## To do
- [x] 4. Tool environment, traces, tests, scripted runs, real-model attempts (done: 42 tests, 10/10 scripted, base 0/20 and SFT 0/20 on tool episodes; see docs/TOOL-ENVIRONMENT.md)
- [x] 5. GRPO speed test (65 s for 10 updates) and bounded run (32 updates, 204 s). Reward signal stayed flat; see docs/GRPO.md
- [x] 6. Evaluate GRPO on validation: 23/200 (SFT 25/200), no improvement; see docs/GRPO.md
- [x] 7. Freeze configs (commit a1a3e2e), held-out test run once for all five models, intervals in RESULTS.md
- [x] 8. RESULTS.md, DATA-CARD.md, MODEL-CARD.md
- [x] 9. Architecture diagram (README), docs/WALKTHROUGH.md, five-minute demo (python -m preference_lab.demo)
- [x] 10. docs/FINAL-STATUS.md; clean-checkout check passed (fresh clone: 43 tests, audit clean)

## Optional, only if time remains
- [ ] Prompt-injection robustness slice
- [ ] Oracle-filter vs random-removal arm on the corrupted data
- [ ] Small read-only viewer for saved predictions

## Not planned
- Calibration (ECE, Brier): the plan says to omit it unless probabilities are defensible.
- Trained multi-turn tool use: only if the trainer supports it, and the plan says to stop after 60 minutes.
